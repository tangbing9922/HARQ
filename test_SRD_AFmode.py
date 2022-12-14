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

def AF_SRD_SS_test(args, SNR, model, StoT):
    test_data = EurDataset("test")
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
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
                    target = sentence
                    sentence = sentence.to(device)
                    src_mask = (sentence == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
                    enc_output = model.encoder(sentence, src_mask)
                    channel_enc_output = model.channel_encoder(enc_output)
                    Tx_sig = PowerNormalize(channel_enc_output)
                    # S --> R channel   高SNR
                    if args.channel == 'AWGN_Relay':
                        RelayReceive_sig = channels.AWGN_Relay(Tx_sig, snr)
                    elif args.channel == 'AWGN_Direct':
                        RelayReceive_sig = channels.AWGN_Direct(Tx_sig, snr)
                    else:
                        raise ValueError("Please choose from AWGN, Rayleigh")
                    # R --> D channel without Decode
                    if args.channel == 'AWGN_Relay':
                        RelaySend_sig = channels.AWGN_Relay(RelayReceive_sig, snr)
                    elif args.channel == 'AWGN_Direct':
                        RelaySend_sig = channels.AWGN_Direct(RelayReceive_sig, snr)
                    else:
                        raise ValueError("Please choose from AWGN, Rayleigh")
                    memory = model.channel_decoder(RelaySend_sig)

                    outputs = torch.ones(src.size(0), 1).fill_(start_idx).type_as(sentence.data)
                    # torch.tensor.fill_(x)用指定的值x填充张量
                    # torch.tensor.type_as(type) 将tensor的类型转换为给定张量的类型
                    for i in range(args.MAX_LENGTH - 1):
                        trg_mask = (outputs == pad_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
                        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
                        combined_mask = torch.max(trg_mask, look_ahead_mask)
                        combined_mask = combined_mask.to(device)

                        # decode the received signal
                        dec_output = model.decoder(outputs, memory, combined_mask, None)
                        pred = model.predict(dec_output)

                        # predict the output_sentences
                        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

                        # return the max-prob index
                        _, next_word = torch.max(prob, dim=-1)
                        outputs = torch.cat([outputs, next_word], dim=1)

                        out_sentences = outputs.cpu().numpy().tolist()
                    for n in range(len(out_sentences)):
                        s = StoT.sequence_to_text(out_sentences[n])
                        out_result_string.append(s)
                    output_sentences.append(out_result_string)
                    target_sent = target.cpu().numpy().tolist()
                    tgt_result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_sentence.append(tgt_result_string)
                    embeddings_output = sentence_model.encode(output_sentence, convert_to_tensor=True)
                    embeddings_target = sentence_model.encode(target_sentence, convert_to_tensor=True)
                    cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
                    for i in range(len(target_sentence)):
                        avg_cos += cos_sim[i][i]
                    avg_cos = avg_cos / len(target_sentence)
                    eachSNR_avg_cos += avg_cos
                eachSNR_avg_cos = eachSNR_avg_cos / len(test_iterator)
                eachSNR_avg_cos_float = eachSNR_avg_cos.cpu().numpy()
                semantic_score.append(eachSNR_avg_cos_float)
            finnal_score.append(semantic_score)
            print("sentence similarity score:", np.mean(finnal_score, axis=0))
            return finnal_score



if __name__ == "__main__":
    setup_seed(7)
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, "r"))
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
    SNR = [4,5,6,7,8,9]
    AF_SRD_SS_test(args, SNR, model, StoT)
