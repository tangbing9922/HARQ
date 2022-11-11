# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/31 10:22”
多跳 语义转发 模式 的性能分析 test 文件
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, initNetParams, semantic_block_train_step, SeqtoText, train_mi, DENSE, greedy_decode, BleuScore
from utils import create_masks
from dataset import EurDataset, collate_data
from Model import DeepTest
from models.transceiver import Cross_Attention_layer, Encoder, Cross_Attention_DeepSC_1026
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess_text import tokenize
from sentence_transformers import SentenceTransformer, util
from train_cross1025 import greedy_decode4cross

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--Relay_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Relay', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=1, type=int)

def multi_jump_test(model, num_jump:int, args, SNR, StoT):
    model.eval()
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    score = []
    score1 = []
    test_data = EurDataset("test")
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
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
                    trg_inp = target[:, :-1]
                    trg_real = target[:, 1:]
                    src_mask, look_ahead_mask = create_masks(target, trg_inp, pad_idx)

                    SR_channel = 'AWGN_Relay'
                    memory = target
                    for i in range(num_jump):
                        out = greedy_decode(SR_model, memory, noise, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)
                        memory = out
                    sentences = out.cpu().numpy().tolist()
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
    pretrained_Relay_checkpoint = torch.load(args.Relay_checkpoint_path + '/1101DeepTest_net_checkpoint.pth')

    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)


    SR_model.load_state_dict(pretrained_Relay_checkpoint['model'])

    sentences = ['the events taking place today in school are extremely interesting']
    #the next item is the joint debate on the following reports

    # single_sentence_test(cross_SC, SR_model, SD_model, sentences)
    SNR = [0, 3, 6, 9, 12, 15, 18]
    multi_jump_test(SR_model, 1, args, SNR, StoT)
    #[0.67789531 0.84526688 0.88982504 0.90228607 0.9082146  0.9111156 0.91311351]
    #bleu4
    #[0.31338311 0.6252166  0.72729268 0.75794071 0.77089684 0.77631573 0.77799713]
    multi_jump_test(SR_model, 2, args, SNR, StoT)
    #[0.53812144 0.79767937 0.87765943 0.89850922 0.90813592 0.90908199 0.91161545]
    #[0.15085767 0.52240363 0.68710042 0.73509233 0.7560983  0.76408734 0.76841077]
    multi_jump_test(SR_model, 3, args, SNR, StoT)
    #[0.43855959 0.75294054 0.8612294  0.8916952  0.90392502 0.90892645 0.91121802]
    #[0.08129941 0.44708516 0.6535119  0.7188505  0.74393219 0.75592423 0.76184188]
    multi_jump_test(SR_model, 4, args, SNR, StoT)
    #[0.37088496 0.71732856 0.84698587 0.88666722 0.90045347 0.90676618 0.90989938]
    #[0.04695291 0.39022219 0.623327   0.70397378 0.73685087 0.74980391 0.75826757]
