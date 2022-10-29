# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/26 9:06”
单句 解码 cross attention测试
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
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_CrossModel', type=str)
parser.add_argument('--Relay_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Relay', type=str)
parser.add_argument('--Direct_checkpoint_path', default='./checkpoints/Train_SemanticBlock_Direct', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=2, type=int)


def single_sentence_test(model, relay_model, direct_model, sentences):
    model.eval()
    relay_model.eval()
    direct_model.eval()

    SNR = 6
    noise_std = SNR_to_noise(SNR)
    noise_std_SD = SNR_to_noise(0)
    noise_std_SR = SNR_to_noise(6)
    with torch.no_grad():
        word = []
        target_word = []
        results = []
        listtest = []
        for sentence in sentences:
            tokens = [2 for _ in range(args.MAX_LENGTH)]
            words = tokenize(sentence, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
            tokens[0:len(words)] = [token_to_idx[word] for word in words]
            results.append(tokens)
            target = torch.tensor(results)
            target = target.to(device)

            trg_inp = target[:, :-1]
            trg_real = target[:, 1:]
            src_mask, look_ahead_mask = create_masks(target, trg_inp, pad_idx)

            SD_channel = 'AWGN_Direct'
            SR_channel = 'AWGN_Relay'
            SD_output = greedy_decode(direct_model, target, noise_std_SD, args.MAX_LENGTH, pad_idx, start_idx, SD_channel)
            SR_output = greedy_decode(relay_model, target, noise_std_SR, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)
            RD_output = greedy_decode(relay_model, SR_output, noise_std_SR, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)

            SD_enc_output = SD_model.encoder(SD_output, src_mask)
            SR_enc_output = SR_model.encoder(RD_output, src_mask)
            out = greedy_decode4cross(model, target, SD_enc_output,SR_enc_output, args.MAX_LENGTH,
                                      pad_idx, start_idx)
            out_sentences = out.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, out_sentences))

        word = word + result_string
        target_word = sentences
        print("target sentence:",target_word)
        print("reconstruct sentence:",word)

def batch_sentence_test_BLEU(model, SR_model, SD_model, args, SNR, StoT):
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
                noise = SNR_to_noise(snr)
                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    trg_inp = target[:, :-1]
                    trg_real = target[:, 1:]
                    src_mask, look_ahead_mask = create_masks(target, trg_inp, pad_idx)

                    SD_channel = 'AWGN_Direct'
                    SR_channel = 'AWGN_Relay'
                    noise_std_SD = SNR_to_noise(0)
                    SD_output = greedy_decode(SD_model, target, noise_std_SD, args.MAX_LENGTH, pad_idx, start_idx, SD_channel)
                    SR_output = greedy_decode(SR_model, target, noise, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)
                    RD_output = greedy_decode(SR_model, SR_output, noise, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)

                    SD_enc_output = SD_model.encoder(SD_output, src_mask)
                    SR_enc_output = SR_model.encoder(RD_output, src_mask)

                    out = greedy_decode4cross(model, target, SD_enc_output, SR_enc_output, args.MAX_LENGTH,
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
    pretrained_Relay_checkpoint = torch.load(args.Relay_checkpoint_path + '/1024DeepTest_net_checkpoint.pth')
    pretrained_Direct_checkpoint = torch.load(args.Direct_checkpoint_path + '/1024DeepTest_net_checkpoint.pth')
    cross_checkpoint = torch.load(args.checkpoint_path + '/1026cross_SC_net_checkpoint_1028.pth')

    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)
    SD_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)

    SR_model.load_state_dict(pretrained_Relay_checkpoint['model'])
    SD_model.load_state_dict(pretrained_Direct_checkpoint['model'])

    cross_SC = Cross_Attention_DeepSC_1026(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    cross_SC.load_state_dict(cross_checkpoint['model'])

    sentences = ['the events taking place today in school are extremely interesting']
    #the next item is the joint debate on the following reports

    # single_sentence_test(cross_SC, SR_model, SD_model, sentences)
    SNR = [0, 3, 6, 9, 12, 15, 18]
    batch_sentence_test_BLEU(cross_SC, SR_model, SD_model, args, SNR, StoT)
    #[0.41507163 0.57712322 0.6193147  0.62849452 0.6330635  0.6340417 0.63516121] Cross attention SC
    #[0.41803028 0.58722765 0.63648964 0.65079835 0.65588512 0.65957994 0.66025217]Cross attention SC 1026
    #[0.37937346 0.52731952 0.56833083 0.57898284 0.58267165 0.58539657 0.58566599]1027
    #[0.41236484 0.57524287 0.61975223 0.63189181 0.63598056 0.63781245 0.63876518]1026_1027 学习率不一样
    #