# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/8/30 10:02”

用于 测试 S --> R --> D 的 AF 模式的模型性能
即 中继 仅仅 转播 信号
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, SeqtoText, DENSE, greedy_decode, BleuScore, Channel_With_PathLoss, PowerNormalize
from dataset import EurDataset, collate_data
from Model import DeepTest
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_SemanticBlock', type=str)
parser.add_argument('--channel', default='AWGN_Relay', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int) #考虑不同的 head数 和 layer数
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=2, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')

def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers设置生成随机数的种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def SRD_SS_test(args, SNR, model, StoT, channel):
    test_data = EurDataset("test")
    test_iterator = DataLoader()
    channels = Channel_With_PathLoss()
    with torch.no_grad():
        for epoch in range(args.epochs):
            output_sentences = []
            target_sentences = []
            semantic_score = []
            for snr in tqdm(SNR):
                # snr = torch.int32(snr)
                SR_noise_std = SNR_to_noise(snr)
                RD_noise_std = SNR_to_noise(snr)  # 不太符合, 就两个信道的噪声随机且不一定方差一样
                eachSNR_avg_cos = 0
                for sentence in test_iterator:
                    out_result_string = []
                    src_mask = (sentence == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
                    enc_output = model.encoder(sentence, src_mask)
                    Tx_sig = PowerNormalize(channel_enc_output)
                    # S --> R channel
                    if channel == 'AWGN_Relay':
                        RelayReceive_sig = channels.AWGN_Relay(Tx_sig, snr)
                    elif channel == 'AWGN_Direct':
                        RelayReceive_sig = channels.AWGN_Direct(Tx_sig, snr)
                    else:
                        raise ValueError("Please choose from AWGN, Rayleigh")
                    # R --> D channel
                    if channel == 'AWGN_Relay':
                        Relaysend_sig = channels.AWGN_Relay(RelayReceive_sig, snr)
                    elif channel == 'AWGN_Direct':
                        Relaysend_sig = channels.AWGN_Direct(RelayReceive_sig, snr)
                    else:
                        raise ValueError("Please choose from AWGN, Rayleigh")
                    memory = model.channel_decoder(Relaysend_sig)


if __name__ == "__main__":
    setup_seed(7)
    args = parser.parse_args()
    vocab = json.load(open(args.vocab, "r"))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    # 模型加载 只在D端进行解码 中继仅仅转发
    model = DeepTest(args.num_layers, num_vocab, num_vocab, args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    model_checkpoint = torch.load('./checkpoints/Train_SemanticBlock/0727DeepTest_net_checkpoint.pth')
    model.load_state_dict(model_checkpoint["model"])
