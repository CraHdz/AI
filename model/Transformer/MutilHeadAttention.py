import math
from torch import nn
import torch
from d2l import torch as d2l

class attention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_head, number_hiddens, dropout=None, bias=False,):
        super(attention, self).__init__()

        #输出为[batch_size, number_hiddens]
        self.W_q = nn.Linear(query_size, number_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, number_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, number_hiddens, bias=bias)
        self.W_o = nn.Linear(number_hiddens, number_hiddens, bias=bias)

        self.attention = d2l.DotProductAttention(dropout)

        self.num_head = num_head

    def forward(self, queries, keys, values, valid_len=None, dropout=None):
        #注意力机制
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        queries = self.transpose_qkv(self.W_q(queries), self.num_head)
        keys = self.transpose_qkv(self.W_k(keys), self.num_head)
        values = self.transpose_qkv(self.W_v(values), self.num_head)

        #单头的attention的运算
        # d_k = queries.size(-1)
        # #key.transpose(-2, -1)将key的最后两维矩阵转置，可以进行高维矩阵乘法
        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
        # if valid_len is not None:
        #     scores = scores.masked_fill(valid_len, -1e9)
        # p_atte = nn.functional.softmax(scores)
        #
        # if dropout is not None:
        #     p_atte= nn.functional.dropout(p_atte, dropout)
        #
        # result = torch.matmul(p_atte, values)

        if valid_len is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_len, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_len)

        output_concat = self.transpose_out(output, self.num_head)

        return self.W_o(output_concat)


    def transpose_qkv(self, x: object, num_heads: object) -> object:
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.premute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def transpose_out(self, x, num_heads):
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)