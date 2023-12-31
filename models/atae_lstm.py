# -*- coding: utf-8 -*-
# file: atae-lstm

from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn

from layers.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

        # self.wp = nn.Parameter(torch.Tensor(opt.hidden_dim,  opt.hidden_dim ), requires_grad=True)
        # self.wx = nn.Parameter(torch.Tensor(opt.hidden_dim,  opt.hidden_dim ), requires_grad=True)
        # self.wp = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2), requires_grad=True)
        # self.wx = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2), requires_grad=True)


    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (h_n, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        r = torch.squeeze(torch.bmm(score, h), dim=1)
        # h_n = h_n.squeeze(0)

        # print(r.size())
        # print(h_n.size())
        # exit(0)

        # h_star = torch.tanh(
        #    torch.matmul(self.wp, torch.transpose(r, 0, 1)) +
        #    torch.matmul(self.wx, torch.transpose(h_n, 0, 1)))

        out = self.dense(r)
        return out


class ATAE_BLSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_BLSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = NoQueryAttention(opt.hidden_dim*2 + opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)


    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (h_n, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        r = torch.squeeze(torch.bmm(score, h), dim=1)
        # h_n = h_n.squeeze(0)

        # print(r.size())
        # print(h_n.size())
        # exit(0)

        # h_star = torch.tanh(
        #    torch.matmul(self.wp, torch.transpose(r, 0, 1)) +
        #    torch.matmul(self.wx, torch.transpose(h_n, 0, 1)))

        out = self.dense(r)
        return out


class ATAE_LSTM_TANH(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM_TANH, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

        self.wp = nn.Parameter(torch.Tensor(opt.hidden_dim,  opt.hidden_dim ), requires_grad=True)
        self.wx = nn.Parameter(torch.Tensor(opt.hidden_dim,  opt.hidden_dim ), requires_grad=True)



    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (h_n, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        r = torch.squeeze(torch.bmm(score, h), dim=1)
        h_n = h_n.squeeze(0)

        # print(r.size())
        # print(h_n.size())
        # exit(0)

        h_star = torch.tanh(
            torch.matmul(self.wp, torch.transpose(r, 0, 1)) +
            torch.matmul(self.wx, torch.transpose(h_n, 0, 1)))

        out = self.dense(torch.transpose(h_star, 0, 1))
        return out
