import torch
import torch.nn as nn


class BERT_ASP(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_ASP, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, auxiliary_bert_seq = inputs[0], inputs[1]
        aspect_len = torch.sum(auxiliary_bert_seq != 0, dim=-1)
        hidden_states, pooler_out = self.bert(text_bert_indices)
        hidden_states = self.dropout(hidden_states)     # [16,85,768]
        auxiliary_bert_seq = auxiliary_bert_seq.unsqueeze(-1)   # [16,85,1]
        aspect = auxiliary_bert_seq*hidden_states
        nonzeros_aspect = aspect_len.float()
        aspect = torch.sum(aspect, dim=1)  # [32, 768]
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        pooler_out = pooler_out
        x = aspect+pooler_out
        logits = self.dense(x)
        return logits