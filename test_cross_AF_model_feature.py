# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/11/30 15:44”
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
from models.transceiver import Cross_Attention_layer, Encoder, Cross_Attention_DeepSC_AF_module
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess_text import tokenize
from sentence_transformers import SentenceTransformer, util
from train_cross1025 import greedy_decode4cross, greedy_decode4cross_feature
from train_cross_feature import getFeature_afterChannel

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_CrossModel_AFmodule_Rayleigh', type=str)
parser.add_argument('--Relay_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Rayleigh_Relay', type=str)
parser.add_argument('--Direct_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Rayleigh_Direct', type=str)
parser.add_argument('--channel', default='Rayleigh_Relay', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=1, type=int)


def batch_sentence_test_BLEU(model, SR_model, SD_model, args, SNR, SNR_SD, StoT):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    score = []
    score1 = []
    model.eval()
    SR_model.eval()
    SD_model.eval()
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
                noise_std_SR = SNR_to_noise(snr)
                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    trg_inp = target[:, :-1]
                    trg_real = target[:, 1:]
                    src_mask, look_ahead_mask = create_masks(target, trg_inp, pad_idx)
                    noise_std_SD = SNR_to_noise(SNR_SD)
                    SD_channel = 'Rayleigh_Direct'
                    SR_channel = 'Rayleigh_Relay'

                    SD_Rx_sig = getFeature_afterChannel(SD_model, target, noise_std_SD, args.MAX_LENGTH, pad_idx, start_idx, SD_channel)
                    SR_output = greedy_decode(SR_model, target, noise_std_SR, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)
                    RD_Rx_sig = getFeature_afterChannel(SR_model, SR_output, noise_std_SR, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)

                    SD_Rx_feature = SD_model.channel_decoder(SD_Rx_sig)
                    SR_Rx_feature = SR_model.channel_decoder(RD_Rx_sig)

                    SD_Rx_feature_AF = model.AF_module(SD_Rx_feature, noise_std_SD)
                    SR_Rx_feature_AF = model.AF_module(SR_Rx_feature, noise_std_SR)

                    # 中午改到这里
                    out = greedy_decode4cross_feature(model, target, SD_Rx_feature_AF, SR_Rx_feature_AF, args.MAX_LENGTH,
                                              pad_idx, start_idx)

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
    #1031 更新Relay mode
    pretrained_Relay_checkpoint = torch.load(args.Relay_checkpoint_path + '/1129DeepTest_net_checkpoint.pth')
    pretrained_Direct_checkpoint = torch.load(args.Direct_checkpoint_path + '/1129DeepTest_net_checkpoint.pth')
    cross_checkpoint = torch.load(args.checkpoint_path + '/1129_cross_SC_net_checkpoint_AFmodule.pth')

    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)
    SD_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)

    SR_model.load_state_dict(pretrained_Relay_checkpoint['model'])
    SD_model.load_state_dict(pretrained_Direct_checkpoint['model'])

    cross_SC = Cross_Attention_DeepSC_AF_module(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    cross_SC.load_state_dict(cross_checkpoint['model'])

    # single_sentence_test(cross_SC, SR_model, SD_model, sentences)
    SNR = [0, 3, 6, 9, 12, 15, 18]
    SNR_SD = [0, 3, 6, 9, 12, 15, 18]
    for SNRSD in SNR_SD:
        batch_sentence_test_BLEU(cross_SC, SR_model, SD_model, args, SNR, SNRSD, StoT)