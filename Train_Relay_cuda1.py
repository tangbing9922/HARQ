# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/7/24 15:41”
0827注: 考虑了量化Q 模块的训练文件
"""
# -*- coding: utf-8 -*-

import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, initNetParams, All_train_step, val_step, SeqtoText, train_mi, DENSE
from dataset import EurDataset, collate_data
from Model import DeepTest, Policy, make_policy
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_RelayWithQ_MI_cuda1', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=200, type=int)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')

def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers设置生成随机数的种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True #for what

def train(epoch, args, net1, Q_net, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    # noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    # noise_std_18 = SNR_to_noise(18)
    _snr = torch.randint(0, 9,(1,))
    #噪声 变
    total = 0
    loss_record = []
    total_cos = 0
    total_MI = 0
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net1, mi_net, sents, _snr, pad_idx, mi_opt, args.channel)
            loss, los_cos = All_train_step(net1, sents, sents, _snr, pad_idx, optimizer, Q_opt, criterion, args.channel, start_idx, sentence_model,
                                           StoT, mi_net, Q_net)
            total += loss
            total_MI += mi
            loss_record.append(loss)
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            pbar.set_description(
                'Epoch: {}:CE Loss:{:.3f} MI{:.3f} los_cos:{:.3f}'.format(
                    epoch + 1, loss, mi, los_cos
                )
            )
        else:
            loss, los_cos = All_train_step(net1, sents, sents, _snr, pad_idx, optimizer, Q_opt, criterion, args.channel, start_idx, sentence_model,
                                           StoT, Q_Net=Q_net)
            total += loss
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            loss_record.append(loss)
            pbar.set_description(
                'Epoch:{};  Type:Train; Loss:{:.3f}; los_cos:{:.3f}'.format(
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

    """ define Q_optimizer and loss function """
    deepTest = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    Q_net = DENSE().to(device)
    mi_net = Mine().to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=2e-5)
    SemanticBlock_checkpoint = torch.load('./checkpoints/Train_SemanticBlock/DeepTest_net_checkpoint.pth')
    deepTest.load_state_dict(SemanticBlock_checkpoint['model'])
    mi_checkpoint = torch.load('./checkpoints/Train_SemanticBlock/mi_net_checkpoint.pth')
    mi_net.load_state_dict(mi_checkpoint['model'])
    mi_opt.load_state_dict(mi_checkpoint['optimizer'])
    Q_checkpoint = torch.load('./checkpoints/Train_Q_net/Q_net_checkpoint.pth')
    Q_net.load_state_dict(Q_checkpoint['model'])

    optimizer = torch.optim.Adam(deepTest.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    Q_opt = torch.optim.Adam(Q_net.parameters(), lr=1e-5)


    epoch_record_loss = []
    total_record_loss = []
    total_record_cos = []
    total_MI = []
    for epoch in range(args.epochs):
        start = time.time()
        std_acc = 10
        total_loss, epoch_record_loss, total_cos, MI_info = train(epoch, args, deepTest, Q_net, mi_net)
        #scheduler.step()
        total_record_loss.append(total_loss)
        total_record_cos.append(total_cos)
        total_MI.append(MI_info)
        print('avg_total_loss in 1 epoch:',total_loss)
        print('cos_loss in 1 epoch:',total_cos)
        print('MI_info in 1 epoch:', MI_info)
        if total_loss < std_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            # if epoch % 10 == 0:
            torch.save({
                    'model': deepTest.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/725DeepTest_net_checkpointWithALL.pth')

            torch.save({
                    'model': Q_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/Q_net_checkpoint.pth')

            torch.save({
                    'model': mi_net.state_dict(),
                    'optimizer': mi_opt.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/mi_net_checkpoint.pth')

            std_acc = total_loss