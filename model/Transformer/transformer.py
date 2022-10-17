import copy
import math

import torch
from torch import nn
from MutilHeadAttention import attention

def clone(module, n):
    #返回一个将module复制n次的modulelist
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

#连接层,俩个子层之间都使用了残差链接和归一化
class subLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(subLayerConnection, self).__init__()
        self.dropout = dropout
        self.layernorm = nn.LayerNorm(size)

    def forward(self, x, mask):
        #layernorm(x +  sublayer(x))
        return self.layernorm(x + self.dropout(mask))


#编码器的整体架构
class Encoder(nn.Module):
    def __init__(self, sublayer, layer_number, size):
        super(Encoder, self).__init__()
        self.modulelist = clone(sublayer, layer_number)
        self.layerNorm = nn.LayerNorm(size)

    def forward(self, x, mask):
        for layer in self.modulelist:
            #每个子层的输出为
            x = layer(x, mask)
            # nn.LayerNorm(x.size()[1:].numel())
        #Encoder的输出数据也要使用归一化
        # return self.layerNorm(x)
        return x

class EncoderSubLayer(nn.Module):
    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderSubLayer, self).__init__()
        self.size = size

        self.attention = self_attention
        self.connection1 = subLayerConnection(size, dropout)

        self.feed_forward = feed_forward
        self.connection2 = subLayerConnection(size, dropout)


    def forward(self, x, mask):
        out1 = self.attention(x, x, x, mask)
        self.connection1(x, out1)

        out2 = self.feed_forward(out1)
        self.connection2(out1, out2)


class DecodeerSubLayer(nn.Module):
    def __int__(self, size, mask_self_attention, attention, feed_forward, dropout):
        super(DecodeerSubLayer, self).__int__()

        self.mask_self_attention = mask_self_attention
        self.connection1 = subLayerConnection(size, dropout)
        self.attention = attention
        self.connection2 = subLayerConnection(size, dropout)
        self.feedforward = feed_forward
        self.connection3 = subLayerConnection(size, dropout)

    def forward(self, x, encoder_input, self_mask, encode_mask):
        out1 = self.mask_self_attention(x, x , x , self_mask)
        out1 = self.connection1(x, out1)

        out2 = self.attention(encoder_input, encoder_input, out1, encode_mask)
        out2 = self.connection2(out1, out2)

        out3 = self.feedforward(out2)
        out3 = self.connection3(out3)

        return out3


class Decoder(nn.Module):
    def __init__(self, sublayer, layer_number):
        super(Decoder, self).__init__()
        self.layers = clone(sublayer, layer_number)

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, encoder_mask)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, src_embeding, target_embeding):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.src_embeding = src_embeding
        self.taget_embeding = target_embeding
        self.generator = Generator()

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)


def positionEncoder():
    pass


