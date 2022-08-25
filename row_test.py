'''
2021.12.21
'''
import torch
from Model import DeepTest  #你这里需要改一下
import torch.nn as nn
import argparse
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SeqtoText, greedy_decode, SNR_to_noise
from preprocess_text import tokenize

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_file', default='/europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='/checkpoints/deepSC-allSNR_AWGN_3layers_128d_withMI323/checkpoint_180.pth', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
###########################################################
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    args = parser.parse_args()
    args.vocab_file = "E:/Desktop/tb/coding/code/model_channel_COS_MI" + args.vocab_file
    # args.checkpoint_path = "E:/Desktop/tb/coding/code/model_channel_COS_MI" + args.checkpoint_path
    args.checkpoint_path = "E:/Desktop/tb/coding/code/DeepSC-master" + args.checkpoint_path
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

    model.load_state_dict(torch.load(args.checkpoint_path))
    #只保存了state_dict时的加载 需要写成model.load_state_dict(torch.load(path))
    sentences = ['the events taking place today in school are extremely interesting']
    # sentences = ['this is the only way to enforce the strict ban on all aid not covered by the code.']
    # sentences = ['the events taking place today in austria are extremely serious.']
    # sentences = ['the events taking place today in school are extremely interesting']

    StoT = SeqtoText(token_to_idx,start_idx, end_idx)
    model.eval()
    SNR9 = SNR_to_noise(9)
    SNR0 = SNR_to_noise(0)
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
            out = greedy_decode(model, SNR0, args.MAX_LENGTH, pad_idx, start_idx, args.channel,, target
            # print(out)
            out_sentences = out.cpu().numpy().tolist()
            # print(out_sentences)
            result_string = list(map(StoT.sequence_to_text, out_sentences))

        word = word + result_string
        target_word = sentences
        print(target_word)
        print(word)