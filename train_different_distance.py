# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/11/8 16:26”
训练 不同距离 的 模型
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import initNetParams, SeqtoText, train_mi, SNR_to_noise
from dataset import EurDataset, collate_data
from Model import DeepTest
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_SemanticBlock_Direct', type=str)
parser.add_argument('--channel', default='AWGN_Direct', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=200, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')


class Channel_with_diff_dis():
    def __init__(self):
        self.device = torch.device('cuda:0')

    def AWGN_Relay(self, Tx_sig, noise_std, distance = 120):#更改
        shape = Tx_sig.shape
        # dim = Tx_sig.shape[0] + Tx_sig.shape[1] + Tx_sig.shape[2]
        # spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        path_loss_exp = -2
        d_ref = 100
        PL = (distance / d_ref) ** path_loss_exp
        Tx_sig = (Tx_sig * PL)
        Tx_sig = Tx_sig.view(Tx_sig.shape[0], -1, 2)
        Tx_sig = Tx_sig + torch.normal(0., noise_std, size=Tx_sig.shape).to(self.device)
        Tx_sig = Tx_sig.view(shape)
        return Tx_sig

    def AWGN_Direct(self, Tx_sig, noise_std, distance = 160):
        shape = Tx_sig.shape
        # dim = Tx_sig.shape[0] + Tx_sig.shape[1] + Tx_sig.shape[2]
        # spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)  # 何时考虑
        path_loss_exp = -2.2  # 路损因子调成多少合适
        d_ref = 100
        PL = (distance / d_ref) ** path_loss_exp
        Tx_sig = Tx_sig * PL
        # std_no = ((10 ** (- SNR / 10.) / 2) ** 0.5).to(self.device) #新增.to(self.device)
        Tx_sig = Tx_sig.view(Tx_sig.shape[0], -1, 2)
        Tx_sig = Tx_sig + torch.normal(0., noise_std, size=Tx_sig.shape).to(self.device)
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


def semantic_block_train_step(model, src, trg, noise_std, pad, opt, criterion, channel, start_symbol, senten_model, S2T, mi_net, distance):
    model.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    # channels = Channel_With_PathLoss_cuda1()
    channels = Channel_with_diff_dis()
    opt.zero_grad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN_Relay':
        Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std, distance)
    elif channel == 'AWGN_Direct':
        Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std, distance)
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


def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers设置生成随机数的种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(epoch, args, net1, mi_net):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    # _snr = torch.randint(-2, 4,(1,))    # 修改, 信道条件较差
    #1031修改 在各种信道下训练
    total = 0
    loss_record = []
    total_cos = 0
    total_MI = 0
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    noise_std = noise_std.astype(np.float64)[0]
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net1, mi_net, sents, noise_std, pad_idx, mi_opt, args.channel)
            loss, los_cos = semantic_block_train_step(net1, sents, sents,noise_std, pad_idx, optimizer, criterion, args.channel, start_idx,
                                                      sentence_model, StoT, mi_net)
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
            loss, los_cos = semantic_block_train_step(net1, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel, start_idx,
                                                      sentence_model, StoT)
            total += loss
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            loss_record.append(loss)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.3f}; los_cos: {:.3f}'.format(
                    epoch + 1, loss,los_cos
                )
            )
    return total / len(train_iterator), loss_record, total_cos / len(train_iterator), total_MI/ len(train_iterator)


if __name__ == '__main__':
    setup_seed(7)
    args = parser.parse_args()
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    deepTest = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepTest.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=2e-5)#改小了学习率
    #初始化与否
    initNetParams(deepTest)
    epoch_record_loss = []
    total_record_loss = []
    total_record_cos = []
    total_MI = []
    for epoch in range(args.epochs):
        start = time.time()
        std_acc = 10
        total_loss, epoch_record_loss, total_cos, MI_info = train(epoch, args, deepTest, mi_net)
        #without MI
        # total_loss, epoch_record_loss, total_cos, _ = train(epoch, args, deepTest)
        total_record_loss.append(total_loss)
        total_record_cos.append(total_cos)
        total_MI.append(MI_info)
        print('avg_total_loss in 1 epoch:',total_loss)
        print('cos_loss in 1 epoch:',total_cos)
        print('MI_info in 1 epoch:', MI_info)
        if total_loss < std_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            if epoch % 10 == 0:
                torch.save({
                    'model': deepTest.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/1107DeepTest_net_checkpoint.pth')

                torch.save({
                    'model': mi_net.state_dict(),
                    'optimizer': mi_opt.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/1107mi_net_checkpoint.pth')

            std_acc = total_loss