# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/25 20:43”
有互信息和语义相似度的cross model的训练
"""
# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/24 21:05”

用于 训练 交叉注意力模块

固定 S --> D SNR = 0dB
S --> R --> D SNR = 9dB
"""
import os
import math
import json
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from utils import SNR_to_noise, SeqtoText, subsequent_mask, Channel_With_PathLoss, loss_function
from utils import initNetParams, create_masks, PowerNormalize, train_mi, greedy_decode
from models.transceiver import Cross_Attention_layer, Encoder, Cross_Attention_DeepSC
from Model import DeepTest
from dataset import EurDataset, collate_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--Relay_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Relay', type=str)
parser.add_argument('--Direct_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Direct', type=str)
parser.add_argument('--saved_checkpoint_path', default='./checkpoints/Train_CrossModel', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=100, type=int)

# 传进来之前 把 relay_model 和 dest_model的encoder的参数复制给 cross_model 对应的两个encoder
# def crossAtten_train_step(relay_model, dest_model, src, trg, noise_std, pad, opt,
#                           criterion, channel, start_symbol, senten_model, S2T, mi_net=None)

def crossAtten_train_step(model, model_SR, model_SD, src, trg, noise_std_SR, noise_std_SD, pad, opt,
                          criterion, start_symbol, senten_model, S2T, mi_net):
    # 在训练中 冻住 model.Relay_encoder 和 Direct_encoder的参数
    model.train()
    model_SD.eval()
    model_SR.eval()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channel_With_PathLoss()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    SD_channel = 'AWGN_Direct'
    SR_channel = 'AWGN_Relay'
    SD_output = greedy_decode(model_SD, src, noise_std_SD, args.MAX_LENGTH, pad, start_symbol, SD_channel)
    SR_output = greedy_decode(model_SR, src, noise_std_SR, args.MAX_LENGTH, pad, start_symbol, SR_channel)

    RD_output = greedy_decode(model_SR, SR_output, noise_std_SR, args.MAX_LENGTH, pad, start_symbol, SR_channel)

    SD_enc_output = model_SD.encoder(SD_output, src_mask)
    SR_enc_output = model_SR.encoder(RD_output, src_mask)

    cross_feature = model.Cross_Attention_Block(SR_enc_output, SD_enc_output, src_mask)

    channel_enc_output = model.channel_encoder(cross_feature)

    channel_dec_output = model.channel_decoder(channel_enc_output)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    target = src
    target_string = target.cpu().numpy().tolist()
    target_sentences = list(map(S2T.sequence_to_text, target_string))
    # greedy_decode得重写
    out = greedy_decode4cross(model, src, SR_enc_output, SD_enc_output, args.MAX_LENGTH, pad, start_symbol)
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
    return loss.item()

def train_Cross(epoch, args, cross_net, SR_net, SD_net, mi_net = None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std_SD = SNR_to_noise(0)
    noise_std_SR = SNR_to_noise(6)

    total = 0
    loss_record = []
    for sents in pbar:
        sents = sents.to(device)
        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net1, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            loss = crossAtten_train_step(cross_net, SR_net, SD_net,
                                sents, sents, noise_std_SR, noise_std_SD,
                                pad_idx, optimizer, criterion, start_idx, StoT, mi_net)
            # MI 和 semantic block 一块训练
            total += loss
            total_MI += mi
            loss_record.append(loss)
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            pbar.set_description(
                'Epoch:{};Type:Train; Loss: {:.3f}; MI{:.3f};los_cos:{:.3f}'.format(
                    epoch + 1, loss, mi, los_cos
                )
            )
        else:
            loss = crossAtten_train_step(cross_net, SR_net, SD_net,
                                sents, sents, noise_std_SR, noise_std_SD,
                                pad_idx, optimizer, criterion, start_idx, StoT, mi_net)
            total += loss
            # los_cos = los_cos.cpu().detach().numpy()
            # total_cos += los_cos
            # total_cos = float(total_cos)
            loss_record.append(loss)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.3f}'.format(
                    epoch + 1, loss
                )
            )
    return total / len(train_iterator), loss_record


def greedy_decode4cross(model, src, direct_feature, relay_feature,
                        max_len, padding_idx, start_symbol, Q_Net=None):
    # create src_mask
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    cross_feature = model.Cross_Attention_Block(relay_feature, direct_feature, src_mask)
    channel_enc_output = model.channel_encoder(cross_feature)
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    memory = model.channel_decoder(channel_enc_output)
    # torch.tensor.fill_(x)用指定的值x填充张量
    # torch.tensor.type_as(type) 将tensor的类型转换为给定张量的类型
    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)

        # predict the output_sentences
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


