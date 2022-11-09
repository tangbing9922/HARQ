# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/27 9:57”
对S->D的模型性能进行测试
"""
# !usr/bin/env python
# -*- coding:utf-8 _*-
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from models.transceiver import Cross_Attention_DeepSC, Cross_Attention_DeepSC_1027, Cross_Attention_DeepSC_1026
from Model import DeepTest

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data32.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/Train_dis_SemanticBlock_Direct/1107DeepTest_net_checkpoint.pth', type=str)
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


def performance(args, SNR, net):
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

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

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

    # model_paths = []
    # for fn in os.listdir(args.checkpoint_path):
    #     if not fn.endswith('.pth'): continue
    #     idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
    #     model_paths.append((os.path.join(args.checkpoint_path, fn), idx))
    #
    # model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    #
    # model_path, _ = model_paths[-1]
    model_path = args.checkpoint_path
    checkpoint = torch.load(model_path)
    deepsc_direct.load_state_dict(checkpoint['model'])
    print('model load!')

    bleu_score = performance(args, SNR, deepsc_direct)
    print(bleu_score)

    #similarity.compute_similarity(sent1, real)
    #[0.47126111 0.46186365 0.45589658 0.44647044 0.44552202 0.4407542 0.44265609]直接链路的性能
    #[0.3872618  0.56218327 0.66083461 0.70544055 0.72354934 0.73184637 0.73522919]1102

    #[0.22120379 0.2584527  0.26804157 0.26761027 0.26756117 0.26806095 0.2687165]1107