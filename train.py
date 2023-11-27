# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import random
import sys
import time
from time import strftime, localtime
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
from transformers import BertModel
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, TNet_LF, AOA, MGAN, ASGCN, RAM_BERT
from models import LCF_BERT, LCF_BERT_ONLY_LOCAL, LCF_BERT_ONLY_GLO, LCF_BERT_LOCAL_CLS, LCF_BERT_ONLY_Global_1
from models import LCF_BERT_xGlo_1_xLocal
from models import LCF_BERT_Attention
from models.aen import AEN_BERT, CrossEntropyLoss_LSR
from models.atae_lstm import ATAE_LSTM, ATAE_BLSTM, ATAE_LSTM_TANH
from models.bert_atae import ATAE_BERT
from models.bert_spc import BERT_SPC
from models.bert_simple import BERT_SIMPLE
from models.lstm import LSTM, BiLSTM
from Loss.SupConLoss import SupConLoss
from Loss.Triplet_Hard import TripletLoss
from Loss.TripletLoss import HardTripletLoss
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def have_different_sample(targets):
    flag = 0
    target1 = targets[0]
    for i in range(len(targets)):
        a = targets[i]
        if a != target1:
            flag = 1
            break
    return flag


def has_opposite_labels(labels):
    return not (labels.sum().item() <= 1 or (1 - labels).sum().item() <= 1)


def plot_jasons_lineplot(x, y, x_label, y_label, title, output_png_path):
    if x == None:
        x = range(1, len(y) + 1)

    _, ax = plt.subplots()
    plt.plot(x, y, linewidth=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=400)
    plt.clf()

    print(f"plot saved at {output_png_path}")


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('building embedding...')
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
            self.pretrained_bert_state_dict = bert.state_dict()
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading dataset...')
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        # self.valset = ABSADataset(opt.dataset_file['val'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params = 0
        n_nontrainable_params = 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'
                    .format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        print('resetting parameters...')
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            else:
                # self.model.load_state_dict(
                #     torch.load("/home/jye/ABSA/laptops_1mio_ep30/laptops_1mio_ep30/pytorch_model.bin"), strict=False)
                self.model.load_state_dict(self.pretrained_bert_state_dict, strict=False)

    def _train(self, CE, TL, SCL, optimizer, train_data_loader, val_data_loader):
        print('staring train...')
        max_val_acc, max_val_f1, max_val_epoch = 0, 0, 0
        global_step = 0
        update_num_list, train_loss_list, train_acc_list, val_acc_list = [], [], [], []
        path = None
        Path(
            f"plots/{self.opt.train_mode}/{self.opt.model_name}/{self.opt.dataset}/{self.opt.local_context_focus}"
            f"/SCL:{self.opt.lamda} TL:{self.opt.beta}/{self.opt.pos_weight, self.opt.neu_weight, self.opt.neg_weight}/{self.opt.seed}") \
            .mkdir(parents=True, exist_ok=True)
        writer = open(
            f"plots/{self.opt.train_mode}/{self.opt.model_name}/{self.opt.dataset}/{self.opt.local_context_focus}"
            f"/SCL:{self.opt.lamda} TL:{self.opt.beta}/{self.opt.pos_weight, self.opt.neu_weight, self.opt.neg_weight}/{self.opt.seed}/logs.csv",
            "w")

        w_tensorboard_log = SummaryWriter(
            comment='{}-{}-{}-{}-{}'.format(self.opt.model_name, self.opt.dataset, self.opt.train_mode, self.opt.margin, self.opt.seed))

        for i_epoch in tqdm(range(self.opt.num_epoch)):
            start = time.time()
            step = 0
            logger.info('>' * 100)
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                logic, sentence_features = self.model(inputs)
                sentiment_labels = batch['polarity'].to(self.opt.device)
                implicit_sentiments = batch['implicit_label'].to(self.opt.device)

                # calculate loss
                loss = 0
                loss1 = CE(logic, sentiment_labels)
                if self.opt.train_mode == 'CE':
                    loss = loss1
                elif self.opt.train_mode == 'CE+TL':
                    if have_different_sample(sentiment_labels):
                        sentence_features = F.normalize(sentence_features, p=2, dim=1)
                        loss2 = TL(sentence_features, sentiment_labels)
                    else:
                        loss2 = 0
                    loss = loss1 + self.opt.beta * loss2
                    # print('loss1:{},loss2:{}'.format(loss1, loss2))
                elif self.opt.train_mode == 'CE+SCL':
                    if not has_opposite_labels(implicit_sentiments):
                        loss3 = 0
                    else:
                        normed_cls_hidden = F.normalize(sentence_features, dim=-1)
                        loss3 = SCL(normed_cls_hidden.unsqueeze(1), labels=implicit_sentiments)
                    loss = loss1 + self.opt.lamda * loss3

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(logic, -1) == sentiment_labels).sum().item()
                n_total += len(logic)
                loss_total += loss.item() * len(logic)
                if step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    val_acc, val_f1 = self._val_evaluate_acc_f1(val_data_loader)
                    logger.info(
                        '[epoch %2d] [step %3d] train_loss: %.4f train_acc: %.4f val_acc: %.4f val_f1: %.4f'
                        % (i_epoch, step, train_loss, train_acc, val_acc, val_f1))
                    w_tensorboard_log.add_scalar('train_loss', train_loss, global_step=global_step)
                    w_tensorboard_log.add_scalar('train_acc', train_acc, global_step=global_step)
                    w_tensorboard_log.add_scalar('val_acc', val_acc, global_step=global_step)
                    w_tensorboard_log.add_scalar('val_f1', val_f1, global_step=global_step)

                    update_num_list.append(global_step)
                    val_acc_list.append(val_acc)
                    train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    writer.write(f"{global_step},{val_acc:.4f},{train_acc:.4f}\n")

                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        save_path = "state_dict/{0}/{1}/{2}/{3}/{4}/" \
                            .format(self.opt.train_mode, self.opt.dataset, self.opt.model_name,
                                    self.opt.local_context_focus, self.opt.seed)
                        dirname = os.path.dirname(save_path)
                        if not os.path.exists(dirname):
                            os.makedirs(dirname)

                        path = save_path + 'epoch{0}_step{1}_acc_{2}_f1{3}' \
                            .format(i_epoch, step, round(val_acc, 4), round(val_f1, 4))
                        if i_epoch > 0:
                            # timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
                            torch.save(self.model.state_dict(), path)
                            logger.info('>> saved: {}'.format(path))
                    if val_f1 > max_val_f1:
                        max_val_f1 = val_f1

            end = time.time()
            logger.info('time: {:.4f}s'.format(end - start))
            if i_epoch - max_val_epoch >= self.opt.patience:
                logger.info('>> early stop.')
                break

            '''==========draw picture========='''
            plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                 f"{self.opt.model_name}  {self.opt.dataset}  {self.opt.train_mode}  max val acc={max(val_acc_list):.3f}",
                                 f"plots/{self.opt.train_mode}/{self.opt.model_name}/{self.opt.dataset}/{self.opt.local_context_focus}"
                                 f"/SCL:{self.opt.lamda} TL:{self.opt.beta}/{self.opt.pos_weight, self.opt.neu_weight, self.opt.neg_weight}/{self.opt.seed}/train_loss.png")
            plot_jasons_lineplot(update_num_list, val_acc_list, 'updates', 'validation accuracy',
                                 f"{self.opt.model_name}  {self.opt.dataset}  {self.opt.train_mode}  max val acc={max(val_acc_list):.3f}",
                                 f"plots/{self.opt.train_mode}/{self.opt.model_name}/{self.opt.dataset}/{self.opt.local_context_focus}"
                                 f"/SCL:{self.opt.lamda} TL:{self.opt.beta}/{self.opt.pos_weight, self.opt.neu_weight, self.opt.neg_weight}/{self.opt.seed}/val_acc{max(val_acc_list):.3f}.png")
            plot_jasons_lineplot(update_num_list, train_acc_list, 'updates', 'train accuracy',
                                 f"{self.opt.model_name}  {self.opt.dataset}  {self.opt.train_mode}  max val acc={max(val_acc_list):.3f}",
                                 f"plots/{self.opt.train_mode}/{self.opt.model_name}/{self.opt.dataset}/{self.opt.local_context_focus}"
                                 f"/SCL:{self.opt.lamda} TL:{self.opt.beta}/{self.opt.pos_weight, self.opt.neu_weight, self.opt.neg_weight}/{self.opt.seed}/train_acc.png")

        logger.info('>> max_val_acc: {:.4f}, max_val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
        return path

    def _val_evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        # self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs, _ = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def _test_evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        predict_wrong = 0
        Neu2Neg, Neu2Pos, Neg2Pos, Neg2Neu, Pos2Neu, Pos2Neg = 0, 0, 0, 0, 0, 0
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs, _ = self.model(t_inputs)
                n_predict = torch.argmax(t_outputs, -1)
                n_correct += (n_predict == t_targets).sum().item()
                n_total += len(t_outputs)
                for i in range(len(t_batch['text'])):
                    if not n_predict[i] == t_targets[i]:
                        predict_wrong += 1
                        logger.info('=' * 40 + 'wrong' + '=' * 40)
                        logger.info(
                            '{0} \n>>>aspect:{1}\n>>>ground_truth: {2}\n>>>predict：{3}'.format(
                                t_batch['text'][i], t_batch['aspect'][i], t_batch['polarity'][i] - 1, n_predict[i] - 1))
                        if t_targets[i] == 1 and n_predict[i] == 0:
                            Neu2Neg += 1
                        elif t_targets[i] == 1 and n_predict[i] == 2:
                            Neu2Pos += 1
                        elif t_targets[i] == 0 and n_predict[i] == 2:
                            Neg2Pos += 1
                        elif t_targets[i] == 0 and n_predict[i] == 1:
                            Neg2Neu += 1
                        elif t_targets[i] == 2 and n_predict[i] == 1:
                            Pos2Neu += 1
                        elif t_targets[i] == 2 and n_predict[i] == 0:
                            Pos2Neg += 1

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        logger.info('[total:{0}] [total predict wrong:{1}] [Neu2Neg:{2}] [Neu2Pos:{3}] [Neg2Pos:{4}] [Neg2Neu:{5}]'
                    ' [Pos2Neu:{6}] [Pos2Neg:{7}]'.format(n_total, predict_wrong, Neu2Neg, Neu2Pos, Neg2Pos, Neg2Neu,
                                                          Pos2Neu, Pos2Neg))
        return acc, f1

    def run(self):
        # Loss and Optimizer
        class_weight = torch.tensor([self.opt.neg_weight, self.opt.neu_weight, self.opt.pos_weight]).to(self.opt.device)
        CE = nn.CrossEntropyLoss(class_weight)
        TL = HardTripletLoss(opt=self.opt, margin=self.opt.margin, hardest=False)
        SCL = SupConLoss(self.opt)
        # criterion = CrossEntropyLoss_LSR(device=self.opt.device, para_LSR=0.2)

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(CE, TL, SCL, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        print(best_model_path)
        test_acc, test_f1 = self._test_evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        logger.info('>> best_model_path: {}'.format(best_model_path))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', default='CE+TL', type=str, help='CE,CE+SCL,CE+TL,CE+SCL+TL')
    parser.add_argument('--model_name', default='lcf_bert_attention', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop, MAMS')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-3, type=float)
    parser.add_argument('--margin', default=0.2, type=float, help='triplet loss margin')
    parser.add_argument('--beta', default=1, type=float, help='triplet loss weight')
    parser.add_argument('--lamda', default=1, type=float, help='SupConLoss weight')
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default="bert-base-uncased", type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--device', default='cuda:5', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=12, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.2, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # parser.add_argument('--cross_val_fold', default=10, type=int, help='k-fold cross validation')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=0, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--pos_weight', default=1.0, type=float)
    parser.add_argument('--neu_weight', default=1.0, type=float)
    parser.add_argument('--neg_weight', default=1.0, type=float)

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'bilstm': BiLSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'atae_bilstm': ATAE_BLSTM,
        'atae_lstm_tanh': ATAE_LSTM_TANH,
        'atae_bert': ATAE_BERT,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'bert_simple': BERT_SIMPLE,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        'lcf_bert_only_lcf': LCF_BERT_ONLY_LOCAL,
        'lcf_bert_only_glo': LCF_BERT_ONLY_GLO,
        'lcf_bert_only_global': LCF_BERT_ONLY_Global_1,
        'lcf_bert_local_cls': LCF_BERT_LOCAL_CLS,
        'lcf_bert_xglo_1_xlocal': LCF_BERT_xGlo_1_xLocal,
        'lcf_bert_attention': LCF_BERT_Attention,
        'ram_bert': RAM_BERT,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            # 'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/train_enhanced.txt",
            # 'train': "/home/disk2/jye/ABSA/datasets/semeval14/few-shot-16.txt",
            # 'val': "/home/disk2/jye/ABSA/datasets/semeval14/Restaurants_Test_Gold.xml.seg",
            # 'train': "/home/disk2/jye/ABSA/datasets/semeval14/Restaurants_Train.xml.seg",
            # 'test': "/home/disk2/jye/ABSA/datasets/semeval14/Restaurants_Test_Gold.xml.seg"
            'train': "/home/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/train_total_text.txt",
            'test': "/home/jye/ABSA/datasets/implicit/restaurant/txt_with_implicit_label4/test_total_text.txt",
            # 'train': "/home/jye/ABSA/datasets/implicit/restaurant/txt1/train_total_text.txt",
            # 'test': "/home/jye/ABSA/datasets/implicit/restaurant/txt1/test_total_text.txt",
            # 'train':"/home/jye/ABSA/datasets/semeval14/Restaurants_Train.xml.seg",
            # 'test':"/home/jye/ABSA/datasets/semeval14/Restaurants_Test_Gold.xml.seg"
            # 'train': "/home/jye/ABSA/datasets/implicit/restaurant/txt/train_total_text.txt",
            # 'test': "/home/jye/ABSA/datasets/implicit/restaurant/txt/test_total_text.txt"

        },
        'laptop': {
            # 'train': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/train_total_text.txt",
            # 'val': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/test_total_text.txt",
            # 'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/test_implicit_text.txt",
            # 'train': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/train_total_text.txt",
            # 'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/test_total_text.txt"
            'train': "/home/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/train_total_text.txt",
            'test': "/home/jye/ABSA/datasets/implicit/laptop/txt_with_implicit_label4/test_total_text.txt",
            # 'train':"/home/jye/ABSA/datasets/semeval14/Laptops_Train.xml.seg",
            # 'test':"/home/jye/ABSA/datasets/semeval14/Laptops_Test_Gold.xml.seg"
            # 'train': "/home/jye/ABSA/datasets/implicit/laptop/txt/train_total_text.txt",
            # 'test': "/home/jye/ABSA/datasets/implicit/laptop/txt/test_total_text.txt"
        },
        'MAMS': {
            'train': "/home/jye/ABSA/datasets/implicit/MAMS/sentence_sentiment_labeled/train.txt",
            'val': "/home/jye/ABSA/datasets/implicit/MAMS/sentence_sentiment_labeled/val.txt",
            'test': "/home/jye/ABSA/datasets/implicit/MAMS/sentence_sentiment_labeled/test.txt"
        },
    }
    input_colses = {
        'lstm': ['text_indices'],
        'bilstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'atae_bilstm': ['text_indices', 'aspect_indices'],
        'atae_lstm_tanh': ['text_indices', 'aspect_indices'],
        'atae_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'ram_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'ram_bert_1': ['text_bert_indices', 'auxiliary_bert_seq', 'aspect_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        # 'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_simple': ['text_bert_indices'],
        'bert_asp': ['text_bert_indices', 'auxiliary_bert_seq'],
        'bert_NLI': ['NLI_bert_indices', 'NLI_segment_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert_only_lcf': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                              'aspect_bert_indices'],
        'lcf_bert_only_glo': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                              'aspect_bert_indices'],
        'lcf_bert_only_global': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                                 'aspect_bert_indices'],
        'lcf_bert_local_cls': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                               'aspect_bert_indices'],
        'lcf_bert_xglo_1_xlocal': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                                   'aspect_bert_indices'],
        'lcf_bert_attention': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices',
                               'aspect_bert_indices'],
        'lcf_attention': ['text_indices', 'aspect_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{0}-train:{1}-{2}-{3}-{4}-{5}-{6}-{7}-({8},{9})-({10},{11},{12})-{13}-{14}.log' \
        .format(opt.train_mode, opt.model_name, opt.dataset, opt.local_context_focus, opt.lr,
                opt.l2reg, opt.batch_size, opt.dropout, opt.lamda, opt.beta, opt.pos_weight, opt.neu_weight,
                opt.neg_weight, opt.seed, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
