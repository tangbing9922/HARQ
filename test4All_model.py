# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/7/25 9:05”
用于语义模块 + 量化 + 路损模型的 测试
"""
import torch
from Model import DeepTest, DENSE
import torch.nn as nn
import argparse
import numpy as np
import os
import json
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SeqtoText, greedy_decode, SNR_to_noise
from preprocess_text import tokenize

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_RelayWithQ_MI', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
###########################################################
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]#2
    model = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    model_checkpoint = torch.load(os.path.join(args.checkpoint_path, 'DeepTest_net_checkpointWithALL.pth'))
    model.load_state_dict(model_checkpoint['model'])
    Q_net = DENSE().to(device)
    # Q_checkpoint = torch.load(os.path.join(args.checkpoint_path, 'Q_net_checkpoint_withALL.pth')) #end2end
    Q_checkpoint = torch.load(os.path.join(args.checkpoint_path, 'Q_net_checkpoint.pth'))
    Q_net.load_state_dict(Q_checkpoint['model'])
    # mi_net = Mine().to(device)
    # mi_net.load_state_dict(torch.load(os.path.join(args.checkpoint_path, '/mi_net_checkpoint_withALL.pth')))

    sentences = ['the events taking place today in school are extremely interesting']
    StoT = SeqtoText(token_to_idx,start_idx, end_idx)
    model.eval()
    Q_net.eval()

    SNR = 8
    with torch.no_grad():
        word = []
        target_word = []
        results = []
        listtest = []
        for sentence in sentences:
            tokens = [2 for _ in range(args.MAX_LENGTH)]#pad 操作
            words = tokenize(sentence, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
            tokens[0:len(words)] = [token_to_idx[word] for word in words]
            results.append(tokens)
            target = torch.tensor(results)
            target = target.to(device)
            out = greedy_decode(model, target, SNR, args.MAX_LENGTH, pad_idx, start_idx, args.channel, Q_net)
            # print(out)
            out_sentences = out.cpu().numpy().tolist()
            # print(out_sentences)
            result_string = list(map(StoT.sequence_to_text, out_sentences))

        word = word + result_string
        target_word = sentences
        print(target_word)
        print(word)