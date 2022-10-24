# -*- coding: utf-8 -*-
"""
2021.12.22
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information
from dataset import EurDataset, collate_data
from sentence_transformers import  util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_len = 32

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

sign = LBSign.apply


class DENSE(nn.Module):
    def __init__(self):
        super(DENSE, self).__init__()
        self.layer1 = nn.Linear(16, 30)
        self.layer2 = nn.Linear(30, 16)

    def Q(self, x):
        return sign(self.layer1(x))

    def dQ(self, x):
        return self.layer2(x)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `learning rate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary,start_idx, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        self.start_idx = start_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            elif idx != self.start_idx:#修改了 不加start token
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words)
    #0829新增 text2idx 方法(batch_size), 不需要, greedy_decode 出来的就能用
    # def text_to_idx(self, list_of_text):
    #     idx_list = []
    #     idx_list.append(self.start_idx)
    #     for sentence in list_of_text:
    #         for word in sentence:
    #             idx_list.append(vocb_dictionary[word])

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)# np.sqrt(2 * snr)的话 进channel的n_var就不用/根号2

    return noise_std



class Channels():
    def __init__(self, device):
        self.device = device

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(self.device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(self.device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(self.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(self.device)
        H_imag = torch.normal(mean, std, size=[1]).to(self.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

class Channel_With_PathLoss():
    def __init__(self):
        self.device = torch.device('cuda:0')

    def AWGN_Relay(self, Tx_sig, noise_std, distance = 100):
        shape = Tx_sig.shape
        # dim = Tx_sig.shape[0] + Tx_sig.shape[1] + Tx_sig.shape[2]
        # spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        path_loss_exp = -2
        d_ref = 100
        PL = (distance / d_ref) ** path_loss_exp
        Tx_sig = (Tx_sig * PL)
        Tx_sig = Tx_sig.view(Tx_sig.shape[0], -1, 2)
        Tx_sig = Tx_sig + torch.normal(0., noise_std, size=Tx_sig.shape).to(device)
        Tx_sig = Tx_sig.view(shape)
        return Tx_sig

    def AWGN_Direct(self, Tx_sig, SNR, distance = 120):
        shape = Tx_sig.shape
        # dim = Tx_sig.shape[0] + Tx_sig.shape[1] + Tx_sig.shape[2]
        # spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)  # 何时考虑
        path_loss_exp = -3
        d_ref = 100
        PL = (distance / d_ref) ** path_loss_exp
        Tx_sig = Tx_sig * PL
        # std_no = ((10 ** (- SNR / 10.) / 2) ** 0.5).to(self.device) #新增.to(self.device)
        std_no = SNR_to_noise(SNR)
        Tx_sig = Tx_sig.view(Tx_sig.shape[0], -1, 2)
        Tx_sig = Tx_sig + torch.normal(0., std_no, size=Tx_sig.shape).to(device)
        Tx_sig = Tx_sig.view(shape)
        return Tx_sig

    def Rayleigh_Relay(self, Tx_sig, SNR, distance = 600):
        shape = Tx_sig.shape
        dim = Tx_sig.shape[0] * Tx_sig.shape[1] * Tx_sig.shape[2]
        spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        path_loss_exp = -2
        d_ref  = 10
        PL = (distance / d_ref) ** path_loss_exp
        coe = ((torch.normal(0, PL**0.5, (Tx_sig.shape[0],1))**2 + torch.normal(0, PL**0.5, (Tx_sig.shape[0],1)) ** 2) ** 0.5) / (2 ** 0.5)
        std_no = (10 ** (- SNR / 10.) / 2) ** 0.5
        Tx_sig = Tx_sig * coe.view(-1, 1, 1).to(self.device)
        Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no * spow
        Tx_sig = Tx_sig.view(shape).to(self.device)
        return Tx_sig

    def Rayleigh_Direct(self, Tx_sig, SNR, distance = 1000):
        if distance == 1000:#或者范围
            path_loss_exp = -3
        shape = Tx_sig.shape
        dim = Tx_sig.shape[0] * Tx_sig.shape[1] * Tx_sig.shape[2]
        spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        d_ref = 10
        PL = (distance / d_ref) ** path_loss_exp
        coe =  ((torch.normal(0, PL**0.5, (Tx_sig.shape[0],1))**2 + torch.normal(0, PL**0.5, (Tx_sig.shape[0],1))**2) ** 0.5) / (2 ** 0.5)
        std_no = (10 ** (- SNR / 10.) / 2) ** 0.5
        Tx_sig = Tx_sig * coe.view(-1, 1, 1)
        Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no * spow
        Tx_sig = Tx_sig.view(shape).to(self.device)
        return Tx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵 decoder 中的 look ahead 操作
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # np.triu 产生上三角矩阵 右上是1， 因为k=1 即主对角线全为 0，再往右上全1 ;k = 0 表示从主对角线开始右上均为1(本例)
    # mask中 True的位置是mask的位置
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    # pad_idx = 0 vocab中设定
    #[batch, 1, seq_len] 128 * 1 * 32
    # 是pad 的位置 置为1， 维度变换成 batch * 1 * seq_len 为了 匹配 batch * seq_len * seq_len ?
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    #[batch, 1, seq_len-1] 128 * 1 * 31
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    # tensor.size(-1)返回最后一个维度的尺寸
    # look_ahead_mask : 1 * 31 * 31
    #
    combined_mask = torch.max(trg_mask, look_ahead_mask)    #torch.max() 返回两个input tensor的最大value及shape
    #combined_mask : 128 * 31 * 31
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)    # criterion选CE cross entropy , x : n * word_dict_size , trg: n,
    mask = (trg != padding_idx).type_as(loss.data)
    #Tensor.type_as(tensor) → Tensor 将此张量强制转换为给定张量的类型。
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)  # 各点信号能量的计算(数字信号中信号的能量即各点信号幅值平方后求和)
    power = torch.mean(x_square).sqrt()
    # torch.mean(x) 返回x所有元素的平均值, 即返回平均能量(总能量/信号长度)
    # 不加sqrt是信号的平均功率, 即sigpower, 也就是方差
    # 加sqrt是因为这是产生对应的噪声标准差
    if power > 1:
        x = torch.div(x, power)
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)    #信号功率设定为1, power_normalize之后能这么理解吗?

    return noise_std

def All_train_step(model, src, trg, snr, pad, opt, Q_opt, criterion, channel, start_symbol, senten_model, S2T, mi_net=None, Q_Net= None):

    model.train()
    Q_Net.train()
    trg_inp = trg[:, :-1]#取 除了最后一列的所有值  不要end token，也不一样，有的长度不够最后是pad的
    trg_real = trg[:, 1:]#取除了第一列的值 不要start token
    # 128 * 31
    channels = Channel_With_PathLoss()
    opt.zero_grad()
    Q_opt.zero_grad()
    #定义一个 Q_opt
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    # src 和 tgt 一致， trg_inp 少了最后一列(why)
    # src_mask 128 * 1 * 32 , look_ahead_mask 128 * 31 * 31
    # encoder 的 src_mask 和 decoder 的 look_ahead_mask
    # look_ahead_mask 是 上三角绝阵，对角线右上全为1(不包括对角)
    # src_mask 是 pad 出来的位置是1

    enc_output = model.encoder(src, src_mask)   #   enc_output : 128 * 32 * 128 batch_size * max_len * d_model
    channel_enc_output = model.channel_encoder(enc_output)  #channel_enc_output: 128 * 32 * 16
    output = PowerNormalize(channel_enc_output)
    Tx_sig = Q_Net.Q(output)

    if channel == 'AWGN_Relay':
        Rx_sig1 = channels.AWGN_Relay(Tx_sig, snr)
    elif channel == 'AWGN_Direct':
        Rx_sig1 = channels.AWGN_Direct(Tx_sig, snr)
    elif channel == 'Rician':
        Rx_sig1 = channels.Rician(Tx_sig, snr)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    Rx_sig = sign(Rx_sig1)
    Rx_sig = Q_Net.dQ(Rx_sig)
    channel_dec_output = model.channel_decoder(Rx_sig)  # 128 * 32 * 128
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)  #dec_output :128 * 31 * 128
    pred = model.predict(dec_output)  # pred :128 * 31 * 23098(Word_dict_size)
    # pred.contiguous().view(-1 , ntokens) --> (128*31 = 3968, 23098) 即 (3968, 23098)
    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1) # word_dict_size
    
    #y_est = x +  torch.matmul(n, torch.inverse(H))
    #loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # 关于 masked 的loss 的计算
    # Tensor.contiguous() 深拷贝tensor内容，两个tensor互不影响，修改pred.contiguous的shape不影响pred
    # 另一种理解 转置、permute、transpose等操作出来的tensor的内存是不连续的，是不contiguous的，所以进行上述操作前期望先深拷贝一下即先分配连续内存
    # trg_real.contiguous().view(-1) 等于把trg_real 做了个flatten : 128 * 31 = 3968 少了第一维 start故31
    #进行贪婪解码 后 计算 cos 的损失 加进 整个loss
    target = src
    target_string = target.cpu().numpy().tolist()
    target_sentences = list(map(S2T.sequence_to_text, target_string))
    out = greedy_decode(model, src, snr, MAX_len, pad, start_symbol, channel, Q_Net)
    out_sentences = out.cpu().numpy().tolist()
    result_sentences = list(map(S2T.sequence_to_text, out_sentences))

    embeddings_target = senten_model.encode(target_sentences, convert_to_tensor=True).to(device)
    embeddings_output = senten_model.encode(result_sentences, convert_to_tensor=True).to(device)
    cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output).to(device)
    # print(cos_sim.shape)
    total_cos = 0
    for i in range(len(target)):
        total_cos += cos_sim[i][i]
    los_cos = total_cos/len(target)
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig1)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb#互信息的相反数
        loss = loss + 0.001 * loss_mine

    loss = 1.5 * loss - 0.5 * los_cos

    loss.backward()#
    clip_gradient(Q_opt, 0.1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    Q_opt.step()
    return loss.item(), los_cos

def semantic_block_train_step(model, src, trg, noise_std, pad, opt, criterion, channel, start_symbol, senten_model, S2T, mi_net=None):
    model.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channel_With_PathLoss()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN_Relay':
        Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std)
    elif channel == 'AWGN_Direct':
        Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std)
    else:
        raise ValueError("Please choose from Relay, Direct")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.predict(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    target = src
    target_string = target.cpu().numpy().tolist()
    target_sentences = list(map(S2T.sequence_to_text, target_string))
    out = greedy_decode(model, src, noise_std, MAX_len, pad, start_symbol, channel)
    out_sentences = out.cpu().numpy().tolist()
    result_sentences = list(map(S2T.sequence_to_text, out_sentences))

    embeddings_target = senten_model.encode(target_sentences, convert_to_tensor=True).to(device)
    embeddings_output = senten_model.encode(result_sentences, convert_to_tensor=True).to(device)
    cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output).to(device)

    total_cos = 0
    for i in range(len(target)):
        total_cos += cos_sim[i][i]
    los_cos = total_cos / len(target)
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.001 * loss_mine

    loss = 1.5 * loss - 0.5 * los_cos

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    return loss.item(), los_cos

def Q_net_train_step(model, src, trg, snr, pad, Q_opt, criterion, channel, Q_Net):
    model.eval()
    Q_Net.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channel_With_PathLoss()
    Q_opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    output = PowerNormalize(channel_enc_output)
    Tx_sig = Q_Net.Q(output)

    if channel == "AWGN_Relay":
        Rx_sig = channels.AWGN_Relay(Tx_sig, snr)
    elif channel == "AWGN_Direct":
        Rx_sig = channels.AWGN_Direct(Tx_sig, snr)
    else:
        raise  ValueError("Please choose from AWGN, Rayleigh")

    Rx_sig = sign(Rx_sig)
    Rx_sig = Q_Net.dQ(Rx_sig)
    loss = criterion(Rx_sig, output)
    loss.backward()
    clip_gradient(Q_opt, 0.1)
    Q_opt.step()

    return loss.item()

def train_mi(model, mi_net, src, noise_std, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()#
    channels = Channel_With_PathLoss()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    #0724
    enc_output = model.encoder(src, src_mask)   #   enc_output : 128 * 32 * 128 batch_size * max_len * d_model
    channel_enc_output = model.channel_encoder(enc_output)  #channel_enc_output: 128 * 32 * 16
    Tx_sig = PowerNormalize(channel_enc_output) # 功率归一化
    if channel == 'AWGN_Relay':
        Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std)
    elif channel == 'AWGN_Direct':
        Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, noise_std)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb#是互信息的相反数

    loss_mine.backward()#
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)#
    opt.step()#

    return loss_mine.item()

#12.22 validate还没改
def val_step(model, Q_Net, src, trg, n_var,pad, criterion, channel, start_symbol):
    model.eval()
    channels = Channels()
    trg_inp = trg[:, :-1]#取 除了最后一列的所有值
    trg_real = trg[:, 1:]#取 除了第一列的所有值

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.predict(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)
    #验证集要不要看cos_loss 感觉无所谓
    
    return loss.item()
    
def greedy_decode(model, src, noise_std, max_len, padding_idx, start_symbol, channel, Q_Net = None):  # greedy中也加入量化模块
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channel_With_PathLoss()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    if Q_Net != None:
        Tx_sig = Q_Net.Q(Tx_sig)
        if channel == 'AWGN_Relay':
            Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std)
        elif channel == 'AWGN_Direct':
            Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh")
        Rx_sig = sign(Rx_sig)
        Rx_sig = Q_Net.dQ(Rx_sig)
    else:
        if channel == 'AWGN_Relay':
            Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std)
        elif channel == 'AWGN_Direct':
            Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh")
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    # torch.tensor.fill_(x)用指定的值x填充张量
    # torch.tensor.type_as(type) 将tensor的类型转换为给定张量的类型
    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.predict(dec_output)
        
        # predict the output_sentences
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs