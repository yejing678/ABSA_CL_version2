import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
from layers.attention import Attention, NoQueryAttention
from layers.squeeze_embedding import SqueezeEmbedding

class ATAE_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(ATAE_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.attention = NoQueryAttention(opt.bert_dim+opt.bert_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.dropout)

        # self.wp = nn.Parameter(torch.Tensor(opt.hidden_dim,  opt.hidden_dim ), requires_grad=True)
        # self.wx = nn.Parameter(torch.Tensor(opt.hidden_dim,  opt.hidden_dim ), requires_grad=True)
        # self.wp = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2), requires_grad=True)
        # self.wx = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2), requires_grad=True)


    def forward(self, inputs):
        text_bert_indices, aspect_bert_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_bert_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_bert_indices != 0, dim=-1).float()

        x, _ = self.bert(text_bert_indices)
        x = self.squeeze_embedding(x, x_len)
        x = self.dropout(x)
        aspect, _ = self.bert(aspect_bert_indices)
        aspect = self.dropout(aspect)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)  # [16, 72, 768]
        ha = torch.cat((x, aspect), dim=-1)
        _, score = self.attention(ha)
        r = torch.squeeze(torch.bmm(score, x), dim=1)
        out = self.dense(r)
        return out



