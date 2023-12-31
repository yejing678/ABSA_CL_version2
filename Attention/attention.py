from torch import nn
import torch.nn.functional as F

INF = 1e9


class Attention(nn.Module):
    """
    The base class of attention.
    """

    def __init__(self, dropout, dim):
        super(Attention, self).__init__()
        self.dropout = dropout
        self.query = nn.Linear(dim, 768)
        self.key = nn.Linear(dim, 768)
        self.value = nn.Linear(dim, 768)
        self.linear = nn.Linear(768, dim)

    def forward(self, query, key, value, mask=None):
        """
        query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        value: FloatTensor (batch_size, time_step, hidden_size)
        mask: ByteTensor (batch_size, time_step) or ByteTensor (batch_size, num_queries, time_step)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        score = self._score(query, key)  # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        # print(weights)
        output = weights.matmul(value)
        if single_query:
            output = output.squeeze(1)
        output = self.linear(output)
        return output

    def _score(self, query, key):
        raise NotImplementedError('Attention score method is not implemented.')

    def _weights_normalize(self, score, mask):
        if not mask is None:
            score = score.masked_fill(mask == 0, INF)
        weights = F.softmax(score, dim=-1)
        return weights

    def get_attention_weights(self, query, key, mask=None):
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        score = self._score(query, key)  # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if single_query:
            weights = weights.squeeze(1)
        return weights
