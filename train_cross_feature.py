# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/29 14:46”
训练 cross_attention 模块 注意 直接链路 和 中继链路  都不解码
特征->信道 ->cross attention
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
# 将 cross_attention的网络参数设置和DeepTest一致
from models.transceiver import Cross_Attention_layer, Encoder, Cross_Attention_DeepSC_1103
from Model import DeepTest
from dataset import EurDataset, collate_data


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
parser.add_argument('--epochs', default=200, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getFeature_afterChannel(model, src, noise_std, max_len, padding_idx, start_symbol, channel):
    model.eval()
    # create src_mask
    channels = Channel_With_PathLoss()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    if channel == 'AWGN_Relay':
        Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std)
    elif channel == 'AWGN_Direct':
        Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")

    return Rx_sig

def crossAtten_train_step(model, model_SR, model_SD, src, trg, noise_std_SR, noise_std_SD, pad, opt,
                          criterion, start_symbol):
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
    # 不用 greedy_decode 用 getFeature_afterChannel
    SD_Rx_sig = getFeature_afterChannel(model_SD, src, noise_std_SD, args.MAX_LENGTH, pad, start_symbol, SD_channel)
    SR_output = greedy_decode(model_SR, src, noise_std_SR, args.MAX_LENGTH, pad, start_symbol, SR_channel)
    RD_Rx_sig = getFeature_afterChannel(model_SR, SR_output, noise_std_SR, args.MAX_LENGTH, pad, start_symbol, SR_channel)

    # 中继链路 S->R 解码 R->D过encoder -> channel encoder -> channel -> channel decoder 之后过cross attention
    # 是否原始模型中就去掉 channel encoder 和 channel decoder，后续实验
    # 1104 先cross 再 model.channel_decoder
    # SD_Rx_feature = model_SD.channel_decoder(SD_Rx_sig)
    # SR_Rx_feature = model_SR.channel_decoder(RD_Rx_sig)

    SD_Rx_feature = model.channel_decoder(SD_Rx_sig)
    SR_Rx_feature = model.channel_decoder(RD_Rx_sig)

    cross_feature = model.Cross_Attention_Block(SR_Rx_feature, SD_Rx_feature, src_mask)
    # cross_output = model.channel_decoder(cross_feature)


    # output = model.nonlinear_transform(cross_feature)
    dec_output = model.decoder(trg_inp, cross_feature, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    return loss.item()

def train_Cross(epoch, args, cross_net, SR_net, SD_net, mi_net = None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std_SD = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    noise_std_SD = noise_std_SD.astype(np.float64)[0]
    noise_std_SR = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))# 每一个epoch 里面变
    noise_std_SR = noise_std_SR.astype(np.float64)[0]
    total = 0
    loss_record = []
    for sents in pbar:
        sents = sents.to(device)
        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net1, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            loss, los_cos = crossAtten_train_step(cross_net, SR_net, SD_net,
                                                      sents, sents, noise_std_SR, noise_std_SD,
                                                      pad_idx, optimizer, criterion, start_idx)
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
                                pad_idx, optimizer, criterion, start_idx)
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



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]  # 2
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    ## 加载预训练 的 Relay model 和 Destination model
    pretrained_Relay_checkpoint = torch.load(args.Relay_checkpoint_path + '/1031DeepTest_net_checkpoint.pth')
    pretrained_Direct_checkpoint = torch.load(args.Direct_checkpoint_path + '/1101DeepTest_net_checkpoint.pth')

    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)
    SD_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)

    SR_model.load_state_dict(pretrained_Relay_checkpoint['model'])
    SD_model.load_state_dict(pretrained_Direct_checkpoint['model'])

    cross_SC = Cross_Attention_DeepSC_1103(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(cross_SC.parameters(),
                                 lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay = 5e-4)

    initNetParams(cross_SC)
    # cross_SC.load_state_dict(pretrained_Relay_checkpoint['model'])

    epoch_record_loss = []
    total_record_loss = []
    total_record_cos = []
    total_MI = []
    for epoch in range(args.epochs):
        start = time.time()
        std_acc = 10
        total_loss, _ = train_Cross(epoch, args, cross_SC, SR_model, SD_model)
        total_record_loss.append(total_loss)
        print('avg_total_loss in 1 epoch:', total_loss)
        if total_loss < std_acc:
            if not os.path.exists(args.saved_checkpoint_path):
                os.makedirs(args.saved_checkpoint_path)
            if epoch % 10 == 0:
                torch.save({
                    'model': cross_SC.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, args.saved_checkpoint_path + '/1104_cross_SC_net_checkpoint_SDrand.pth')
                # cross feature

            std_acc = total_loss