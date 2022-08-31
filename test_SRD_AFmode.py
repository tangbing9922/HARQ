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
from utils import  SNR_to_noise, initNetParams, semantic_block_train_step, SeqtoText, train_mi, DENSE, greedy_decode, BleuScore
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

def SRD_SS_test(args, SNR, net, StoT):
    test_data = EurDataset("test")
    test_iterator = DataLoader()
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
                    tgt_result_string = []
                    target_sentence = []
                    output_sentence = []
                    avg_cos = 0
                    a = sentence.size(0)  # len of each sentence batch
                    sentence = sentence.to(device)
                    target = sentence
                    out_SR = greedy_decode(SR_model, sentence, SR_noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
                    # 考虑noise_std 并不一样
                    out_RD = greedy_decode(RD_model, out_SR, RD_noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
                    out_idx_list = out_RD.cpu().numpy().tolist()
        #             for n in range(len(out_idx_list)):
        #                 s = StoT.sequence_to_text(out_idx_list[n])
        #                 out_result_string.append(s)
        #             output_sentence.append(out_result_string)
        #             target_sent = target.cpu().numpy().tolist()
        #             tgt_result_string = list(map(StoT.sequence_to_text, target_sent))
        #             target_sentence.append(tgt_result_string)
        #             embeddings_output = sentence_model.encode(output_sentence, convert_to_tensor=True)
        #             embeddings_target = sentence_model.encode(target_sentence, convert_to_tensor=True)
        #             cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
        #             for i in range(len(target_sentence)):
        #                 avg_cos += cos_sim[i][i]
        #             avg_cos = avg_cos / len(target_sentence)
        #             eachSNR_avg_cos += avg_cos
        #         eachSNR_avg_cos = eachSNR_avg_cos / len(test_iterator)
        #         eachSNR_avg_cos_float = eachSNR_avg_cos.cpu().numpy()
        #         semantic_score.append(eachSNR_avg_cos_float)
        #     finnal_score.append(semantic_score)
        # print("sentence similarity score:", np.mean(finnal_score, axis=0))
        # return finnal_score
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
