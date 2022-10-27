# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/24 21:05”

用于 训练 交叉注意力模块

固定 S --> D SNR = 0dB
S --> R --> D SNR = 6dB
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
from models.transceiver import Cross_Attention_layer, Encoder, Cross_Attention_DeepSC_1027
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
parser.add_argument('--epochs', default=100, type=int)

# 传进来之前 把 relay_model 和 dest_model的encoder的参数复制给 cross_model 对应的两个encoder
# def crossAtten_train_step(relay_model, dest_model, src, trg, noise_std, pad, opt,
#                           criterion, channel, start_symbol, senten_model, S2T, mi_net=None)

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
    SD_output = greedy_decode(model_SD, src, noise_std_SD, args.MAX_LENGTH, pad, start_symbol, SD_channel)
    SR_output = greedy_decode(model_SR, src, noise_std_SR, args.MAX_LENGTH, pad, start_symbol, SR_channel)
    # noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1)) # noise_std[0]
    RD_output = greedy_decode(model_SR, SR_output, noise_std_SR, args.MAX_LENGTH, pad, start_symbol, SR_channel)

    SD_enc_output = model_SD.encoder(SD_output, src_mask)
    SR_enc_output = model_SR.encoder(RD_output, src_mask)

    cross_feature = model.Cross_Attention_Block(SR_enc_output, SD_enc_output, src_mask)

    # channel_enc_output = model.channel_encoder(cross_feature)
    #
    # channel_dec_output = model.channel_decoder(channel_enc_output)
    output = model.nonlinear_transform(cross_feature)
    dec_output = model.decoder(trg_inp, output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    # target = src
    # target_string = target.cpu().numpy().tolist()
    # target_sentences = list(map(S2T.sequence_to_text, target_string))
    # out = greedy_decode(model, src, noise_std, MAX_len, pad, start_symbol, channel)
    # out_sentences = out.cpu().numpy().tolist()
    # result_sentences = list(map(S2T.sequence_to_text, out_sentences))
    #
    # embeddings_target = senten_model.encode(target_sentences, convert_to_tensor=True).to(device)
    # embeddings_output = senten_model.encode(result_sentences, convert_to_tensor=True).to(device)
    # cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output).to(device)
    #
    # total_cos = 0
    # for i in range(len(target)):
    #     total_cos += cos_sim[i][i]
    # los_cos = total_cos / len(target)

    # if mi_net is not None:
    #     mi_net.eval()
    #     joint, marginal = sample_batch(Tx_sig, Rx_sig)
    #     mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    #     loss_mine = -mi_lb
    #     loss = loss + 0.001 * loss_mine

    # loss = 1.5 * loss - 0.5 * los_cos

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
    pretrained_Relay_checkpoint = torch.load(args.Relay_checkpoint_path + '/1024DeepTest_net_checkpoint.pth')
    pretrained_Direct_checkpoint = torch.load(args.Direct_checkpoint_path + '/1024DeepTest_net_checkpoint.pth')

    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)
    SD_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)

    SR_model.load_state_dict(pretrained_Relay_checkpoint['model'])
    SD_model.load_state_dict(pretrained_Direct_checkpoint['model'])

    cross_SC = Cross_Attention_DeepSC_1027(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(cross_SC.parameters(),
                                 lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay = 5e-4)

    initNetParams(cross_SC)

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
                }, args.saved_checkpoint_path + '/1027cross_SC_net_checkpoint.pth')

            std_acc = total_loss


    # deepsc_dict = deepsc.state_dict()
    # print(pretrained_Relay_checkpoint.items())


    # cross_dict = cross_SC.state_dict()
    # # print(deeptest_dict.keys())
    # for name, params in cross_SC.Relay_encoder.named_parameters():
    #     print(name)
    #
    # for name, params in cross_SC.Relay_encoder.named_parameters():
    #     params_name = 'Relay_encoder.' + name
    #     pretrain_name = 'encoder' + name
    #     cross_SC.state_dict()['params_name'].copy_(pretrained_Relay_checkpoint['model'][pretrain_name])

    # for key_name in pretrained_Relay_checkpoint['model'].keys():
    #     print(key_name)

    # new_dict = {k: v for k, v in pretrained_Relay_dict.items() if k in deepsc_dict.keys()}
    # deepsc_dict.update(new_dict)
    # deepsc.load_state_dict(deepsc_dict)