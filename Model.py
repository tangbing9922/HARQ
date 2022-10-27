import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]#在哪个维度上加 x的第2个维度（x.size(1)）
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0 #务必整除 模型尺寸必须能整除多头数
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads # 128/8 = 16 即 每一个query key value 的维度，最后再把8个头的拼接起来还原为128维
        self.num_heads = num_heads
        #要bias
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


# class LayerNorm(nn.Module):
#    "Construct a layernorm module (See citation for details)."
#    # features = d_model
#    def __init__(self, features, eps=1e-6):
#        super(LayerNorm, self).__init__()
#        self.a_2 = nn.Parameter(torch.ones(features))
#        self.b_2 = nn.Parameter(torch.zeros(features))
#        self.eps = eps
#
#    def forward(self, x):
#        mean = x.mean(-1, keepdim=True)
#        std = x.std(-1, keepdim=True)
#        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        # self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        # m = memory

        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # q, k, v
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, num_layers, src_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)

        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output


class DeepTest(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(DeepTest, self).__init__()

        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256),
                                             # nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))

        self.channel_decoder = ChannelDecoder(16, d_model, 256)# 和deepSC的区别就在这

        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)

        self.predict = nn.Linear(d_model, trg_vocab_size)# 名字改了， 无语


class Policy(nn.Module):#策略网络
    "Implement the PE function."
    def __init__(self, n1, L,device):
        super(Dense2, self).__init__()
        self.layer1 = nn.Linear(n1,1)
        self.layer2 = nn.Linear(L+1,3)
        self.device = device
        self.relu=nn.ReLU()

    def forward(self, x, _snr):
        x=self.layer1(x).to(self.device)
        snr=_snr.to(self.device)
        x=torch.squeeze(x)
        x=self.relu(x)
        x=torch.cat((x,snr),1).to(self.device) #B*L+1
        x=self.layer2(x).to(self.device)  #B*3
        #print(x.shape)
        return x

def make_policy(device,N1=16,N2=31):
    "Helper: Construct a model from hyperparameters."
    model = Policy(N1,N2,device)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

sign = LBSign.apply

class DENSE(nn.Module):#量化网络
    def __init__(self):
        super(DENSE, self).__init__()
        self.layer1 = nn.Linear(16, 30)
        self.layer2 = nn.Linear(30, 16)

    def Q(self, x):
        return sign(self.layer1(x))

    def dQ(self, x):
        return self.layer2(x)