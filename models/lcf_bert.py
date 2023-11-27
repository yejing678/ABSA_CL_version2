# -*- coding: utf-8 -*-
# file: lcf_bert.py
# The code is based on repository: https://github.com/yangheng95/LCF-ABSA
import json

import torch
import torch.nn as nn
import copy
import numpy as np

# from transformers.modeling_bert import BertPooler, BertSelfAttention
from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention, BertPreTrainedModel


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert  # Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        # self.dense1 = nn.Linear(opt.bert_dim, opt.polarities_dim)
        # self.dense2 = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 2
                                        - self.opt.SRD) / np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, bert_spc_pooler = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)
        # bert_spc_pooler = bert_spc_pooler.unsqueeze(dim=1)
        # bert_global_out = torch.cat((bert_spc_pooler, bert_spc_out), dim=1)
        bert_global_out = bert_spc_out

        bert_local_out, bert_local_pooler = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)
        # bert_local_pooler = bert_local_pooler.unsqueeze(dim=1)
        # bert_local_out = torch.cat((bert_local_pooler, bert_local_out), dim=1)

        # a=torch.zeros_like(bert_local_out)
        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            # print(masked_local_text_vec)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            # a=weighted_text_local_features
            # print(weighted_text_local_features)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_global_out, bert_local_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        aspect_logic = self.dense(pooled_out)
        # sentence_logic = self.dense2(pooled_out)

        # a = a.tolist()

        # json.dump(a,open('./tem.json','w'))

        return aspect_logic, pooled_out


class LCF_BERT_ONLY_LOCAL(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT_ONLY_LOCAL, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert  # Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 2
                                        - self.opt.SRD) / np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, bert_spc_pooler = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)
        # bert_spc_pooler = bert_spc_pooler.unsqueeze(dim=1)
        # bert_global_out = torch.cat((bert_spc_pooler, bert_spc_out), dim=1)
        bert_global_out = bert_spc_out

        bert_local_out, bert_local_pooler = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)
        # bert_local_pooler = bert_local_pooler.unsqueeze(dim=1)
        # bert_local_out = torch.cat((bert_local_pooler, bert_local_out), dim=1)

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            # print(weighted_text_local_features)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = bert_local_out
        mean_pool = self.linear_single(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)

        return dense_out, pooled_out


class LCF_BERT_ONLY_Global_1(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT_ONLY_Global_1, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert  # Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 2
                                        - self.opt.SRD) / np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, bert_spc_pooler = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)
        # bert_spc_pooler = bert_spc_pooler.unsqueeze(dim=1)
        # bert_global_out = torch.cat((bert_spc_pooler, bert_spc_out), dim=1)
        bert_global_out = bert_spc_out
        bert_local_out, bert_local_pooler = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)
        # bert_local_pooler = bert_local_pooler.unsqueeze(dim=1)
        # bert_local_out = torch.cat((bert_local_pooler, bert_local_out), dim=1)

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            # print(weighted_text_local_features)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = bert_global_out
        mean_pool = self.linear_single(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)

        return dense_out, pooled_out
