import os
import sys

import torch
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader

from data_utils import build_tokenizer, build_embedding_matrix, ABSADataset

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

'''准备数据'''


def prepare_data(dataset_file, dataset):
    tokenizer = build_tokenizer(
        fnames=[dataset_file['train'], dataset_file['test']],
        max_seq_len=85,
        dat_fname='{0}_tokenizer.dat'.format(dataset))
    embedding_matrix = build_embedding_matrix(
        word2idx=tokenizer.word2idx,
        embed_dim=300,
        dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(300), dataset))
    trainset = ABSADataset(dataset_file['train'], tokenizer)
    testset = ABSADataset(dataset_file['test'], tokenizer)
    train_data_loader = DataLoader(dataset=trainset, shuffle=True)
    test_data_loader = DataLoader(dataset=testset, shuffle=False)
    return train_data_loader, test_data_loader, embedding_matrix


class SVM_features(nn.Module):
    def __init__(self, embedding_matrix):
        super(SVM_features, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

    def forward(self, inputs):
        x_l, x_r, labels = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        x_l_feature = torch.max(x_l, dim=-1)
        x_r_feature = torch.max(x_r, dim=-1)
        feartures = torch.cat((x_l_feature, x_r_feature))
        return feartures, labels


if __name__ == '__main__':
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            # 'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/train_enhanced.txt",
            # 'train': "/home/disk2/jye/ABSA/datasets/semeval14/few-shot-16.txt",
            'val': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/test_total_text.txt",
            'train': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/train_enhanced.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/restaurant/txt1/test_implicit_text.txt"
        },
        'laptop': {
            'train': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/train_total_text.txt",
            'val': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/test_total_text.txt",
            'test': "/home/disk2/jye/ABSA/datasets/implicit/laptop/txt1/test_implicit_text.txt",
        },
        'MAMS': {
            'train': './datasets/MAMS/train.txt',
            'val': './datasets/MAMS/val.txt',
            'test': './datasets/MAMS/test.txt'
        },
        'mooc': {
            'train': "/home/jingy/ABSA/datasets/zh_mooc/mooc.train.txt",
            'test': "/home/jingy/ABSA/datasets/zh_mooc/mooc.test.txt"
        },
        'yelp': {
            'train': "/home/jingy/ABSA/datasets/yelp/yelp.train.txt",
            'test': "/home/jingy/ABSA/datasets/yelp/yelp.test.txt"
        },

    }
    dataset = 'restaurant'
    dataset_file = dataset_files[dataset]
    train_data_loader, test_data_loader, embedding_matrix = prepare_data(dataset_file, dataset)
    inputs_cols = ['left_with_aspect_indices', 'right_with_aspect_indices', 'polarity']
    test_cols = ['left_with_aspect_indices', 'right_with_aspect_indices', 'polarity']
    train_inputs,l1,r1,p1=[],[],[],[]
    for i_batch, batch in enumerate(train_data_loader):
        l1.append(batch['left_with_aspect_indices'])
        r1.append(batch['right_with_aspect_indices'])
        p1.append(batch['polarity'])
    l1 = torch.tensor([item.detach().numpy() for item in l1])
    r1 = torch.tensor([item.detach().numpy() for item in r1])
    p1 = torch.tensor([item.detach().numpy() for item in p1])
    train_inputs.append(l1)
    train_inputs.append(r1)
    train_inputs.append(p1)
    test_inputs,l2, r2, p2= [],[], [], []
    for i_batch, batch in enumerate(test_data_loader):
        l2.append(batch['left_with_aspect_indices'])
        r2.append(batch['right_with_aspect_indices'])
        p2.append(batch['polarity'])
    l2 = torch.tensor([item.detach().numpy() for item in l2])
    r2 = torch.tensor([item.detach().numpy() for item in r2])
    p2 = torch.tensor([item.detach().numpy() for item in p2])
    test_inputs.append(l2)
    test_inputs.append(r2)
    test_inputs.append(p2)
    # test_inputs = torch.tensor([item.detach().numpy() for item in test_inputs]).cuda()
    # test_inputs = torch.tensor([item.numpy() for item in test_inputs])

    # train_inputs = [train_data_loader[col] for col in inputs_cols]
    # test_inputs = [train_data_loader[col] for col in test_cols]
    clf = SVC(C=5, gamma=0.05, max_iter=200)
    svm=SVM_features(embedding_matrix)
    train_features, train_labels = svm(train_inputs)
    test_features, test_labels = svm(test_inputs)
    clf.fit(train_features, train_labels)
    # Test on Training data
    train_result = clf.predict(train_features)
    precision = sum(train_result == train_labels) / train_labels.shape[0]
    print('Training precision: ', precision)

    # Test on test data
    test_result = clf.predict(test_features)
    precision = sum(test_result == test_labels) / test_labels.shape[0]
    print('Test precision: ', precision)
