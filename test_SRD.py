# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/8/27 13:23”

用于测试 信源S -> 中继R -> 信宿D 的模型  (DF模式)
batch 批量测试 BLEU + Simlarity
在测试集上做测试, 单句测试还得重新写.
需要首先加载 从 信源S --> 中继R 的模型
值得一提的是，不能 加载 训好的 S --> R 的模型 来训练 R --> D的模型
因为 S-->R 模型加载之后不应该采用训练集数据作为输入
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
parser.add_argument('--checkpoint_path', default='./checkpoints/Train_SemanticBlock_Rayleigh_Relay', type=str)
parser.add_argument('--channel', default='Rayleigh_Relay', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int) #考虑不同的 head数 和 layer数
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=1, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')

def setup_seed(seed):
    torch.manual_seed(seed)#set the seed for generating random numbers设置生成随机数的种子
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def batch_sentenceSim_test(args, SNR, StoT, SR_model, RD_model):   # 0829下午继续改
    finnal_score = []
    test_data = EurDataset("test")
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    with torch.no_grad():
        for epoch in range(args.epochs):
            output_sentences = []
            target_sentences = []
            semantic_score = []
            for snr in tqdm(SNR):
                # snr = torch.int32(snr)
                SR_noise_std = SNR_to_noise(snr)
                RD_noise_std = SNR_to_noise(snr) #不太符合, 就两个信道的噪声随机且不一定方差一样
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
                    out_RD =greedy_decode(RD_model, out_SR, RD_noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
                    out_idx_list = out_RD.cpu().numpy().tolist()
                    for n in range(len(out_idx_list)):
                        s = StoT.sequence_to_text(out_idx_list[n])
                        out_result_string.append(s)
                    output_sentence.append(out_result_string)
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

def single_sentenceSim_test(args, SR_model, RD_model):  # 还没做测试, 铁不能用
    SR_model.eval()
    RD_model.eval()
    with torch.no_grad():
        target_sentence = ["today is a rainy day"]
        output_word = []
        target_word = []
        results = []
        for sentence in target_sentence:
            tokens = [2 for _ in range(args.MAX_LENGTH)]  # pad 2 max_len
            words = tokenize(sentence, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
            tokens[0:len(words)] = [token_to_idx[word] for word in words]
            results.append(tokens)
            target = torch.tensor(results)
            target = target.to(device)
            out = greedy_decode(model, target, SNR, args.MAX_LENGTH, pad_idx, start_idx, args.channel)  # model 这里要改要加
            out_sentences = out.cpu().numpy().tolist()
            t = np.array(out_sentences)
            print(t.shape)
            result_string = list(map(StoT.sequence_to_text, out_sentences))

        output_word = output_word + result_string
        target_word = sentences

        embeddings_target = sentence_model.encode(target_word, convert_to_tensor=True)
        embeddings_output = sentence_model.encode(output_word, convert_to_tensor=True)

        cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
        total_cos = 0
        print(cos_sim.shape)
        for i in range(len(target_word)):
            print("sen1:{} \t\t sen2:{} \t\t Score: {:.4f}".format(target_word[i], output_word[i], cos_sim[i][i]))
            total_cos += cos_sim[i][i]

        los_cos = total_cos/len(target_word)
        print(los_cos)

def batch_BLEU_test(args, SNR, StoT, SR_model, RD_model):
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
                    SR_out = greedy_decode(SR_model, target, noise, args.MAX_LENGTH, pad_idx, start_idx, args.channel) #改
                    RD_out = greedy_decode(RD_model, SR_out, noise, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
                    sentences = RD_out.cpu().numpy().tolist()
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

    # # 具体的模型路径需要再确定
    # 加载SR_model
    SR_model = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    SR_checkpoint = torch.load('./checkpoints/Train_SemanticBlock_Rayleigh_Relay/1213DeepTest_net_checkpoint.pth')
    SR_model.load_state_dict(SR_checkpoint['model'])

    #加载RD_model
    RD_model = DeepTest(args.num_layers, num_vocab, num_vocab, args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    RD_checkpoint = torch.load('./checkpoints/Train_SemanticBlock_Rayleigh_Relay/1213DeepTest_net_checkpoint.pth')
    RD_model.load_state_dict(RD_checkpoint['model'])

    SNR = [0,3,6,9,12,15,18] # [0,3,6,9,12,15,18] [0,5,10,15,20,25,30,35,40]
    # test_sim_score = batch_sentenceSim_test(args, SNR, StoT, SR_model, RD_model)    # sentence similarity compute
    score = batch_BLEU_test(args, SNR, StoT, SR_model, RD_model)    # BLEU score compute
    # print(score)

    #bleu1 S-R->D
    #[0.53723313 0.80026352 0.87785538 0.8998838  0.90758901 0.91137339 0.91350405]
    #[0.57857619 0.8552709  0.8967281  0.8977264  0.89459219 0.89323794 0.89085762]
    #bleu1 S->D

    #[0.27463424 0.46264492 0.5806899  0.63233463 0.6561114  0.66920263 0.67410422] 1107 SRD
    #[0.14433565 0.19838376 0.26330865 0.3204145  0.42538853 0.48309748 0.56946303]# 没补偿路损
    #[0.25862494 0.50820628 0.59055231 0.66452122 0.75977189 0.70316944 0.79203189]# 没路损
    #[0.14322584 0.20088391 0.26802963 0.3277445  0.43945113 0.50097404 0.59093764]# 把路损补偿了
    #[0.1873226  0.4074056  0.52336228 0.59998559 0.70761992 0.67398394 0.76501392]# 1117 重新训练的补偿了路损的SRD
    # [0.18999291 0.53242364 0.66742076 0.76303501 0.78844015 0.82353547 0.83635673 0.83863477 0.84000533] 0-40dB
    # [0.18156312 0.39696406 0.5135011  0.5886589  0.69547914 0.66373784 0.75767886]
    #  语义DF [0.18156312 0.39696406 0.5135011  0.5886589  0.69547914 0.66373784 0.75767886]

