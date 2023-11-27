import os
import sys
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
import torch.nn.functional as F
from layers.attention import Attention, NoQueryAttention

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class RAM_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(RAM_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        # self.attention = Attention(opt.bert_dim, opt.bert_dim)
        self.att_linear = nn.Linear(opt.bert_dim + opt.bert_dim, 1)
        self.gru_cell = nn.GRUCell(opt.bert_dim, opt.bert_dim)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        left_len = left_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1 - (left_len[i] - idx) / memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i] + aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i] + aspect_len[i], memory_len[i]):
                weight[i].append(1 - (idx - left_len[i] - aspect_len[i] + 1) / memory_len[i])
                u[i].append(idx - left_len[i] - aspect_len[i] + 1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = torch.tensor(u, dtype=memory.dtype).to(self.opt.device).unsqueeze(2)
        weight = torch.tensor(weight).to(self.opt.device).unsqueeze(2)
        v = memory * weight
        memory = torch.cat([v, u], dim=2)
        return memory

    def forward(self, inputs):
        memory, aspect = inputs[0], inputs[1]
        # bert encode
        memory_len = torch.sum(memory != 0, dim=-1)
        aspect_len = torch.sum(aspect != 0, dim=-1)
        memory = self.squeeze_embedding(memory, memory_len)  # [16, 55]
        memory, _ = self.bert(memory)  # [16, 55, 768]
        memory = self.dropout(memory)  # [16, 55, 768]
        aspect = self.squeeze_embedding(aspect, aspect_len)  # [16, 5]
        aspect, _ = self.bert(aspect)  # [16, 5, 768]
        aspect = self.dropout(aspect)  # [16, 5, 768]
        nonzeros_aspect = aspect_len.float()  # [5., 3., 5., 3., 3., 3., 3., 3., 3., 4., 4., 3., 3., 3., 3., 3.]
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))  # [16, 768]

        et = torch.zeros_like(aspect).to(self.opt.device)  # [16, 768]

        batch_size = memory.size(0)
        seq_len = memory.size(1)
        # print((torch.zeros(batch_size, seq_len, self.opt.bert_dim).to(self.opt.device) + et.unsqueeze(1)).size())
        # [16, 55, 768]
        # print(et.unsqueeze(1).size())  # [16, 1, 768]

        for _ in range(self.opt.hops):
            x = torch.cat([memory,
                           torch.zeros(batch_size, seq_len, self.opt.bert_dim).to(self.opt.device) + et.unsqueeze(1) + aspect.unsqueeze(
                               1)],
                          dim=-1)  # [16, 55, 2304]
            # q = et + aspect
            # q = self.dropout(q)
            # q = q.cpu().detach().numpy()
            # memory = memory.cpu().detach().numpy()
            # output, at = Attention(memory, q)
            # output = self.dropout(output)
            g = self.att_linear(x.float())
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), memory.float()).squeeze(1)  # [16, 768]
            i = self.dropout(i)
            et = self.gru_cell(i, et)
            et = self.dropout(et)
            print(alpha)
        out = self.dense(et)
        return out
