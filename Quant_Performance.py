# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/12/7 9:26”
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import QUANT_DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, Quant_greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()

parser.add_argument('--vocab-file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--quant-checkpoint-path', default='checkpoints/Quantization_4bit_Rayleigh_Relay/checkpoint_100.pth', type=str)
parser.add_argument('--channel', default='Rayleigh_Relay', type=str)
parser.add_argument('--MAX-LENGTH', default=32, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=2, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def performance(args, SNR, net):
    bleu_score_1gram = BleuScore(0, 0, 0, 1)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)    # drop_last

    StoT = SeqtoText(token_to_idx, start_idx, end_idx)
    score = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)
                # list4matlab = []
                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    out = Quant_greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)
                    # list4matlab.append(listMat)
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
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))  # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

    score1 = np.mean(np.array(score), axis=0)

    return score1

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = QUANT_DeepSC(args.num_layers, num_vocab, num_vocab,
                          num_vocab, num_vocab, args.d_model, args.num_heads,
                          args.dff, 0.1, num_bit=4).to(device)

    model_paths = []
    is_quant = True
    checkpoint_path = args.quant_checkpoint_path if is_quant else args.checkpoint_path
    print('load path:', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    bleu_score = performance(args, SNR, deepsc)

    print(bleu_score)
    # [0.3501264  0.45638512 0.55767028 0.65622807 0.73002881 0.78329521 0.78755096]
    # [0.27659904 0.34546302 0.47160414 0.5757239  0.65603106 0.71597091 0.75002227]