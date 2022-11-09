# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/11/9 9:24”
用于测试 不同距离 的 S -> D 的 模型性能
"""
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode4difdis, SeqtoText
from tqdm import tqdm
from models.transceiver import Cross_Attention_DeepSC, Cross_Attention_DeepSC_1027, Cross_Attention_DeepSC_1026
from Model import DeepTest

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data32.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/Trian_dis_SemanticBlock_Direct/1108DeepTest_net_checkpoint.pth', type=str)
parser.add_argument('--channel', default='AWGN_Direct', type=str)
parser.add_argument('--MAX-LENGTH', default=32, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=3, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=2, type = int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def performance(args, SNR, net, distance):
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
                    out = greedy_decode4difdis(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel, distance= distance)

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

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc_direct = DeepTest(args.num_layers, num_vocab, num_vocab,
                     args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                     args.dff, 0.1).to(device)

    model_path = args.checkpoint_path
    checkpoint = torch.load(model_path)
    deepsc_direct.load_state_dict(checkpoint['model'])
    print('model load!')
    distance_list = [100,120,140,160,180,200]
    for distance in distance_list:
        bleu_score = performance(args, SNR, deepsc_direct, distance)
        print(bleu_score)

# 200 [0.18666704 0.21409653 0.22474331 0.22754747 0.22667493 0.22563565 0.22462052]
# 100 [0.57733577 0.67331975 0.69315537 0.69482437 0.69445219 0.69081847 0.68956179]