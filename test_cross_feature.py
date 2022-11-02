# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/31 11:46”
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
from train_cross1025 import greedy_decode4cross, greedy_decode4cross_feature
from train_cross_feature import getFeature_afterChannel

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
                noise_std_SR = SNR_to_noise(snr)
                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    trg_inp = target[:, :-1]
                    trg_real = target[:, 1:]
                    src_mask, look_ahead_mask = create_masks(target, trg_inp, pad_idx)
                    noise_std_SD = SNR_to_noise(3)
                    SD_channel = 'AWGN_Direct'
                    SR_channel = 'AWGN_Relay'
                    # 不用 greedy_decode 用 getFeature_afterChannel
                    SD_Rx_sig = getFeature_afterChannel(SD_model, target, noise_std_SD, args.MAX_LENGTH, pad_idx, start_idx, SD_channel)
                    SR_output = greedy_decode(SR_model, target, noise_std_SR, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)
                    RD_Rx_sig = getFeature_afterChannel(SR_model, SR_output, noise_std_SR, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)

                    # 中继链路 S->R 解码 R->D过encoder -> channel encoder -> channel -> channel decoder 之后过cross attention
                    # 是否原始模型中就去掉 channel encoder 和 channel decoder，后续实验
                    SD_Rx_feature = SD_model.channel_decoder(SD_Rx_sig)
                    SR_Rx_feature = SR_model.channel_decoder(RD_Rx_sig)
                    # 中午改到这里
                    out = greedy_decode4cross_feature(model, target, SD_Rx_feature, SR_Rx_feature, args.MAX_LENGTH,
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
    pretrained_Relay_checkpoint = torch.load(args.Relay_checkpoint_path + '/1031DeepTest_net_checkpoint.pth')
    pretrained_Direct_checkpoint = torch.load(args.Direct_checkpoint_path + '/1101DeepTest_net_checkpoint.pth')
    cross_checkpoint = torch.load(args.checkpoint_path + '/1102cross_SC_net_checkpoint_SDrand.pth')

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
    #[0.47289038 0.65106855 0.65806355 0.63884641 0.62660946 0.61965274 0.61611706]1031
    #[0.60500147 0.79472286 0.80953209 0.80281588 0.7964331  0.79383961 0.79146097]1101

    #用只在SD=0dB下训练过的Cross-model来做测试
    #[0.5522777  0.77690512 0.79921402 0.79490276 0.79126624 0.79124564 0.79042571]0dB : SD snr
    #[0.62185511 0.78128252 0.79494959 0.78900196 0.78462144 0.78310721 0.78178828]3dB
    #[0.61152588 0.80617499 0.82298724 0.81500397 0.80923748 0.8072806 0.80513101]6dB
    #[0.61645822 0.8079647  0.82556628 0.8159333  0.81157131 0.80705604 0.80598944]9dB

    # 1102下午SD和SRD都是0-18dB下随机训练的cross-model
    #[0.59451386 0.78736328 0.80584895 0.79875022 0.79421145 0.79070837 0.79014251]0 : SD snr
    #[0.63462385 0.80558392 0.813909   0.80487099 0.79904299 0.79506224 0.79391638]3