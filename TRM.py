import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_heads):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.n_heads = n_heads
        self.d_k = 64
        self.d_v = 64
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_Q.weight)
        init.xavier_normal_(self.W_K.weight)
        init.xavier_normal_(self.W_V.weight)
        init.xavier_normal_(self.linear.weight)


    def forward(self, Q, K, V, attn_mask):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_heads):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, inputs):
        residual = inputs

        output = nn.ReLU()(self.fc1(inputs))
        output = self.fc2(output)

        return self.layer_norm(output + residual)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_ff, n_layers, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, n_layers, n_heads)

    def forward(self, dec_inputs_vetor, dec_self_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs_vetor, dec_inputs_vetor, dec_inputs_vetor, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_heads):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_layers, n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs_vetor, dec_inputs):

        # get_attn_pad_mask 自注意力层的时候的pad 部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask.cuda() + dec_self_attn_subsequent_mask.cuda()), 0)

        for layer in self.layers:
            dec_inputs_vetor = layer(dec_inputs_vetor.cuda(), dec_self_attn_mask.cuda())

        return dec_inputs_vetor


class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_heads):
        super(Transformer, self).__init__()
        self.decoder = Decoder(d_model, d_ff, n_layers, n_heads)


    def forward(self, dec_inputs_vetor, dec_inputs):

        dec_outputs = self.decoder(dec_inputs_vetor, dec_inputs)

        return dec_outputs


