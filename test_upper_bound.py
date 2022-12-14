# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/11/3 20:54”
获取性能上限， 即没有噪声和信道的干扰。
"""
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, upperbound_greedy_decode, SeqtoText
from tqdm import tqdm
from models.transceiver import Cross_Attention_DeepSC, Cross_Attention_DeepSC_1027, Cross_Attention_DeepSC_1026
from Model import DeepTest

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data32.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/Train_SemanticBlock_Rayleigh_Relay/1120DeepTest_net_checkpoint.pth', type=str)
parser.add_argument('--channel', default='Rayleigh_Relay', type=str)
parser.add_argument('--MAX-LENGTH', default=32, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=3, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=2, type = int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def SD_upper_performance(args, SNR, net):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, start_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = upperbound_greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)

    score1 = np.mean(np.array(score), axis=0)
    # score2 = np.mean(np.array(score2), axis=0)

    return score1#, score2

def SRD_upper_performance(args, SNR, StoT, SR_model, RD_model):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    score = []
    score1 = []
    SR_model.eval()
    RD_model.eval()
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
                    SR_out = upperbound_greedy_decode(SR_model, target, noise, args.MAX_LENGTH, pad_idx, start_idx) #改
                    # RD_out = upperbound_greedy_decode(RD_model, SR_out, noise, args.MAX_LENGTH, pad_idx, start_idx)
                    sentences = SR_out.cpu().numpy().tolist()
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
    args = parser.parse_args()
    SNR = [0,5,10,15,20,25,30,35,40]
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    """ define optimizer and loss function """
    deepsc_direct = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)

    model_path = args.checkpoint_path
    checkpoint = torch.load(model_path)
    deepsc_direct.load_state_dict(checkpoint['model'])
    print('model load!')

    # SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
    #                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
    #                     args.dff, 0.1).to(device)
    # SR_checkpoint = torch.load('./checkpoints/Train_SemanticBlock_Relay/1101DeepTest_net_checkpoint.pth')
    # SR_model.load_state_dict(SR_checkpoint['model'])
    #
    # #加载RD_model
    # RD_model = DeepTest(args.num_layers, num_vocab, num_vocab, args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
    #                     args.dff, 0.1).to(device)
    # RD_checkpoint = torch.load('./checkpoints/Train_SemanticBlock_Relay/1101DeepTest_net_checkpoint.pth')
    # RD_model.load_state_dict(RD_checkpoint['model'])

    bleu_score = SD_upper_performance(args, SNR, deepsc_direct)
    # score = SRD_upper_performance(args, SNR, StoT, SR_model, RD_model)
    print(bleu_score)