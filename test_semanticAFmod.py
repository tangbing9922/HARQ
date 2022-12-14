# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/12/7 14:59”
用于语义 AF 模式 中继转发
加载训练好的模型 只是连续过信道 不解码
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, initNetParams, semantic_block_train_step, SeqtoText, train_mi, DENSE, AF_greedy_decode, BleuScore
from dataset import EurDataset, collate_data
from Model import DeepTest
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_SemanticBlock_Rayleigh_Relay', type=str)
parser.add_argument('--channel', default='Rayleigh_Relay', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int) #考虑不同的 head数 和 layer数
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=1, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')

def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def batch_BLEU_test(args, SNR, StoT, SR_model, RD_model):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    score = []
    score1 = []
    SR_model.eval()
    RD_model.eval()
    test_data = EurDataset("test")
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    num_relay = 1
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                output_word = []
                target_word = []
                noise = SNR_to_noise(snr)
                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    AF_SRD_out = AF_greedy_decode(SR_model, target, noise, args.MAX_LENGTH, pad_idx, start_idx, args.channel, num_relay)
                    sentences = AF_SRD_out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    output_word = output_word + result_string
                    target_sent = target.cpu().numpy().tolist()
                    target_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + target_string
                Tx_word.append(target_word)
                Rx_word.append(output_word)
            bleu_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)
    score1 = np.mean(np.array(score), axis=0)
    print(score1)
    return score1

if __name__ == '__main__':
    setup_seed(7)
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    SR_checkpoint = torch.load('./checkpoints/Train_SemanticBlock_Rayleigh_Relay/1129DeepTest_net_checkpoint.pth')
    SR_model.load_state_dict(SR_checkpoint['model'])

    # 加载RD_model
    RD_model  = DeepTest(args.num_layers, num_vocab, num_vocab, args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    RD_checkpoint = torch.load('./checkpoints/Train_SemanticBlock_Rayleigh_Relay/1129DeepTest_net_checkpoint.pth')
    RD_model.load_state_dict(RD_checkpoint['model'])

    SNR = [0,3,6,9,12,15,18]
    score = batch_BLEU_test(args, SNR, StoT, SR_model, RD_model)
    # AF [0.4136409  0.51308868 0.57376147 0.7181762  0.77746216 0.74764802 0.80368039]

