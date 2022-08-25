# -*- coding: utf-8 -*-
"""
2021.12.27
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, initNetParams, All_train_step, val_step, SeqtoText, train_mi
from dataset import EurDataset, collate_data
from Model import DeepTest, DENSE
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/20220722relay', type=str)
parser.add_argument('--MI_model_path', default='./checkpoints/MI_model', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
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
    torch.backends.cudnn.deterministic = True #for what

def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    noise_std_18 = SNR_to_noise(18)
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, noise_std[0], pad_idx,
                            criterion, args.channel, start_idx)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    noise_std_18 = SNR_to_noise(18)
    #噪声 变
    total = 0
    loss_record = []
    total_cos = 0
    total_MI = 0
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net, mi_net, noise_std[0], pad_idx, mi_opt, args.channel, )
            loss, los_cos = All_train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel, start_idx,
                                           sentence_model, StoT)
            total += loss
            total_MI += mi
            loss_record.append(loss)
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}; los_cos: {:.5f}'.format(
                    epoch + 1, loss, mi, los_cos
                )
            )
        else:
            loss, los_cos = All_train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel, start_idx,
                                           sentence_model, StoT)
            total += loss
            los_cos = los_cos.cpu().detach().numpy()
            # print(type(los_cos))#numpy.ndarray
            total_cos += los_cos
            total_cos = float(total_cos)
            # print(type(total_cos))#numpy.float64
            loss_record.append(loss)
            # print(type(total))
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; los_cos: {:.5f}'.format(
                    epoch + 1, loss,los_cos
                )
            )
    return total / len(train_iterator), loss_record, total_cos / len(train_iterator), total_MI/ len(train_iterator)


if __name__ == '__main__':
    setup_seed(10)
    args = parser.parse_args()
    # args.vocab_file = 'E:/Desktop/tb/coding/code/model_channel_COS_MI' + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    # args.num_layers = 3
    # args.d_model = 128
    """ define Q_optimizer and loss function """
    deepTest = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    #先不加互信息
    # print(deepTest)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepTest.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=2e-5)#改小了学习率
    #初始化与否
    initNetParams(deepTest)
    # deepTest.load_state_dict(torch.load('checkpoints/deepTest_cos&MI_32_1230_stdnoise/checkpoint_100.pth'))
    epoch_record_loss = []
    total_record_loss = []
    total_record_cos = []
    total_MI = []
    #epoch_record_loss 是一个epoch中的loss记录（波动比较大）是一个列表
    #total_loss 是一个epoch最后的平均loss   是一个数
    #画 total_loss的图
    for epoch in range(args.epochs):
        start = time.time()
        std_acc = 1
        #train(epoch, args, deepTest)# 没加互信息
        #选择和互信息一块训练
        total_loss, epoch_record_loss, total_cos, MI_info = train(epoch, args, deepTest,mi_net)
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
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepTest.state_dict(), f)
                std_acc = total_loss
    # if not os.path.exists(args.MI_model_path):
    #     os.makedirs(args.MI_model_path)
    # with open(args.MI_model_path + '/MIcheckpoint.pth', 'wb')as f1:
    #     torch.save(mi_net.state_dict(), f1)
    # with open("E:/Desktop/tb/coding/code/model_channel_COS_MI/3_128_trianSNR0_both_100.txt",'w+') as f2:
    #     f2.write(str(total_record_loss))
    # plt.figure(1)
    # plt.title('SNR0 train avg_loss info ')
    # plt.xlabel('Epoch')
    # plt.ylabel('avg_loss')
    # y_major_locator = MultipleLocator(0.2)
    # ax = plt.gca()#ax为两条坐标轴的实例
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.plot(total_record_loss)
    # plt.show()


    plt.figure(2)
    plt.title('SNR0 train avg_cos_loss info ')
    plt.xlabel('Epoch')
    plt.ylabel('avg_cos_loss')
    plt.plot(total_record_cos)
    plt.show()

    plt.figure(3)
    plt.title('SNR0 MI_infom_32')
    plt.xlabel('Epoch')
    plt.ylabel('Avg_MI')
    plt.plot(total_MI)
    plt.show()

    print('All done!')
