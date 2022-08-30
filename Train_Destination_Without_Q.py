# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/7/26 21:20”
S-->D 的模型训练
Destination's semantic block, SNR : [-2 , 3] -2 -1 0 1 2 3  高SNR 4 5 6 7 8 9
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import initNetParams, semantic_block_train_step, SeqtoText, train_mi, DENSE
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
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_Destination_SemanticBlock_withoutQ', type=str)
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

def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers设置生成随机数的种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(epoch, args, net1, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    _snr = torch.randint(-2, 3,(1,))

    total = 0
    loss_record = []
    total_cos = 0
    total_MI = 0
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net1, mi_net, sents, _snr, pad_idx, mi_opt, args.channel)
            loss, los_cos = semantic_block_train_step(net1, sents, sents, _snr, pad_idx, optimizer, criterion, args.channel, start_idx,
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
            loss, los_cos = semantic_block_train_step(net1, sents, sents, _snr, pad_idx, optimizer, criterion, args.channel, start_idx,
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
                }, args.checkpoint_path + '/0727DeepTest_net_checkpoint.pth')

                torch.save({
                    'model': mi_net.state_dict(),
                    'optimizer': mi_opt.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/0727mi_net_checkpoint.pth')

            std_acc = total_loss