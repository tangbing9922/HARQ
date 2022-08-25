# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/7/24 19:22”
训练Q_net
对比Q之前和dQ之后的 MSE
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, initNetParams, Q_net_train_step, SeqtoText, DENSE
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
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_Q_net', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
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

def train(epoch, args, net1, Q_net):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    _snr = torch.randint(0, 9,(1,))
    total = 0
    loss_record = []

    for sents in pbar:
        sents = sents.to(device)
        loss = Q_net_train_step(net1, sents, sents, _snr, pad_idx, Q_optimizer, criterion, args.channel, Q_net)
            # MI 和 semantic block 一块训练
        total += loss
        loss_record.append(loss)
        pbar.set_description(
                'Epoch:{};Type:Train; MSE Loss: {:.3f}'.format(
                    epoch + 1, loss
                )
            )
    return total / len(train_iterator), loss_record


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
    criterion = nn.MSELoss()
    Q_optimizer = torch.optim.Adam(Q_net.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(Q_optimizer, milestones=[10, 20, 30, 40], gamma=0.3)
    SemanticBlock_checkpoit = torch.load("./checkpoints/Train_SemanticBlock/DeepTest_net_checkpoint.pth")
    deepTest.load_state_dict(SemanticBlock_checkpoit['model'])
    epoch_record_loss = []
    total_record_loss = []
    for epoch in range(args.epochs):
        start = time.time()
        std_acc = 10
        total_loss, epoch_record_loss = train(epoch, args, deepTest, Q_net)
        scheduler.step()
        total_record_loss.append(total_loss)
        print('avg_total_loss in 1 epoch:',total_loss)
        if total_loss < std_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            if epoch % 10 == 0:
                torch.save({
                    'model': Q_net.state_dict(),
                    'Q_optimizer': Q_optimizer.state_dict(),
                    'epoch': epoch,
                }, args.checkpoint_path + '/Q_net_checkpoint.pth')

            std_acc = total_loss