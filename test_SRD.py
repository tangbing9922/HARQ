# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/8/27 13:23”

用于测试 信源S -> 中继R -> 信宿D 的模型
在测试集上做测试, 单句测试还得重新写.
需要首先加载 从 信源S --> 中继R 的模型
值得一提的是，不能 加载 训好的 S --> R 的模型 来训练 R --> D的模型
因为S-->R模型加载之后不应该采用训练集数据作为输入

两种思路：
1.两部分模型单独训练，最后evaluate模式下进行整体的模型构建
2.两部分模型同时训练 从S->R->D一块整
思考上述两种方式的区别在哪儿？
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import  SNR_to_noise, initNetParams, semantic_block_train_step, SeqtoText, train_mi, DENSE, greedy_decode
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
parser.add_argument('--epochs', default=200, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')

def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers设置生成随机数的种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(epoch, args, net1, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    _snr = torch.randint(4, 10,(1,))

    total = 0
    loss_record = []
    total_cos = 0
    total_MI = 0
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net1, mi_net, sents, _snr, pad_idx, mi_opt, args.channel)
            loss, los_cos = semantic_block_train_step(net1, sents, sents, _snr, pad_idx, optimizer, criterion, args.channel, start_idx,
                                                      sentence_model, StoT, mi_net)
            # MI 和 semantic block 一块训练
            total += loss
            total_MI += mi
            loss_record.append(loss)
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            pbar.set_description(
                'Epoch:{};Type:Train; Loss: {:.3f}; MI{:.3f};los_cos:{:.3f}'.format(
                    epoch + 1, loss, mi, los_cos)
            )
        else:
            loss, los_cos = semantic_block_train_step(net1, sents, sents, _snr, pad_idx, optimizer, criterion, args.channel, start_idx,
                                                      sentence_model, StoT)
            total += loss
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            loss_record.append(loss)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.3f}; los_cos: {:.3f}'.format(
                    epoch + 1, loss,los_cos
                )
            )
    return total / len(train_iterator), loss_record, total_cos / len(train_iterator), total_MI/ len(train_iterator)


if __name__ == '__main__':
    setup_seed(7)
    args = parser.parse_args()
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)

    # 加载SR_model
    SR_Model = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    mi_checkpoint = torch.load('./checkpoints/Train_SemanticBlock/0727mi_net_checkpoint.pth')
    mi_net.load_state_dict(mi_checkpoint['model'])
    # # 具体的模型路径需要再确定
    SR_checkpoint = torch.load('./checkpoints/Train_Destination_SemanticBlock_withoutQ/0727DeepTest_net_checkpoint.pth')
    SR_Model.load_state_dict(SR_checkpoint['model'])

    #加载RD_model
    RD_model = DeepTest(args.num_layers, num_vocab, num_vocab, args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    RD_checkpoint = torch.load('./checkpoints/Train_Destination_SemanticBlock_withoutQ/0727DeepTest_net_checkpoint.pth')
    RD_model.load_state_dict(RD_checkpoint['model'])

    SR_Model.eval()
    RD_model.eval()

    test_data = EurDataset("test")
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    SNR = [4,5,6,7,8,9]

    with torch.no_grad():
        for epoch in range(args.epochs):
            output_sentences = []
            target_sentences = []
            semantic_score = []
            for snr in tqdm(SNR):
                snr = torch.int32(snr)
                noise_std = SNR_to_noise(snr)
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
                    out = greedy_decode(SR_Model, sentence, noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
                    out_sentence = out.cpu().numpy().tolist()
                    # 解码如何把128个句子分别解码之后再添加到string
                    for n in range(len(out_sentence)):
                        s = StoT.sequence_to_text(out_sentence[n])
                        out_result_string.append(s)
                    # 直接用out_result_string
                    output_sentence.append(out_result_string)
                    target_sent = target.cpu().numpy().tolist()
                    tgt_result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_sentence.append(tgt_result_string)
                    # embeddings_output = sentence_model.encode(output_sentence, convert_to_tensor=True)
                    # embeddings_target = sentence_model.encode(target_sentence, convert_to_tensor=True)
                    # cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
                    for i in range(len(target_sentence)):
                        avg_cos += cos_sim[i][i]
                    avg_cos = avg_cos / len(target_sentence)
                    eachSNR_avg_cos += avg_cos
                eachSNR_avg_cos = eachSNR_avg_cos / len(test_iterator)
                eachSNR_avg_cos_float = eachSNR_avg_cos.cpu().numpy()
                semantic_score.append(eachSNR_avg_cos_float)
            finnal_score.append(semantic_score)
        print("sentence similarity score:", np.mean(finnal_score, axis=0))