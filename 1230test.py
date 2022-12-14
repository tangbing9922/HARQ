# from utils import SeqtoText, greedy_decode, SNR_to_noise
# import torch
# import argparse
# import json
# from preprocess_text import tokenize
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--vocab_file', default='europarl/vocab32.json', type=str)
# parser.add_argument('--checkpoint_path', default='/checkpoints/deepTest_cos&MI_32_1228/checkpoint_85.pth', type=str)
# parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
# parser.add_argument('--MAX_LENGTH', default=32, type=int)
# parser.add_argument('--MIN_LENGTH', default=4, type=int)
# parser.add_argument('--d_model', default=128, type=int)
# parser.add_argument('--dff', default=512, type=int)
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--num_layers', default=4, type=int)
# parser.add_argument('--num_heads', default=8, type=int)
# parser.add_argument('--epochs', default=2, type=int)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# if __name__ == '__main__':
#     args = parser.parse_args()
#     vocab = json.load(open(args.vocab_file, 'rb'))
#     token_to_idx = vocab['token_to_idx']
#     idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
#     num_vocab = len(token_to_idx)
#     pad_idx = token_to_idx["<PAD>"]
#     start_idx = token_to_idx["<START>"]
#     end_idx = token_to_idx["<END>"]#2
#     sentences = ['i am simply asking the commission to come and inform the committee on budgets before implementing the second tranche .','you are my destiny?','today is a sun day so i want to play ball with my friends']
#     StoT = SeqtoText(token_to_idx, start_idx, end_idx)
#     result_idx = []
#     out_result_list = []
#     for sentence in sentences:
#         tokens = [0 for _ in range(args.MAX_LENGTH)]  # pad 操作
#         words = tokenize(sentence, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
#         tokens[0:len(words)] = [token_to_idx[word] for word in words]
#         result_idx.append(tokens)
#     for n in range(len(result_idx)):
#         s = StoT.sequence_to_text(result_idx[n])
#         out_result_list.append(s)
#     print(out_result_list)
#
# #################################################################################
'''
2021.12.21
'''
'''
将这个任务做成两个阶段的任务 
现在计算 句子丢进训练好的模型做相似度计算，以相似度计算作为损失
'''
import torch
from Model import DeepTest
import torch.nn as nn
import argparse
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SeqtoText, greedy_decode, SNR_to_noise
from preprocess_text import tokenize
from sentence_transformers import SentenceTransformer, util
from dataset import EurDataset, collate_data

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints/deepTest_cos&MI_32_1228/checkpoint_85.pth', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--epochs', default=2, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]#2

    args.num_layers = 6
    args.num_heads = 8
    args.d_model = 256
    SNR = [0, 3, 6, 9, 12, 15, 18]
    args.checkpoint_path = '..//model_channel_COS_MI/checkpoints/deepTest_cos&MI_32_1228/checkpoint_100.pth'
    # args.checkpoint_path = 'E:/Desktop/tb/coding/code/DeepSC-master/checkpoints/deepTest-AWGN_withMI1229/checkpoint_99.pth'
    # args.vocab_file = 'checkpoints/deepTest_32_1224/checkpoint_82.pth'#都只训练100个epoch，效果明显好过不加cos
    model = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    sentence_model = SentenceTransformer(
        'models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')
    # model.load_state_dict(torch.load("./checkpoints/deepTest_32_1222/checkpoint_50.pth"))
    model.load_state_dict(torch.load(args.checkpoint_path))
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)
    model.eval()
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    with torch.no_grad():
        for epoch in range(args.epochs):
            semantic_score = []
            for snr in tqdm(SNR):
                noise_std = SNR_to_noise(snr)
                eachSNR_avg_cos = 0
                for sentence in test_iterator:
                    out_result_string = []
                    tgt_result_string = []
                    target_sentence = []
                    output_sentence = []
                    avg_cos = 0
                    a = sentence.size(0)#len of each sentence batch
                    sentence = sentence.to(device)
                    target = sentence
                    ##########################################################################
                    # 需要改
                    out = greedy_decode(model, target, snr, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
                    out_sentence = out.cpu().numpy().tolist()
                    #解码如何把128个句子分别解码之后再添加到string
                    for n in range(len(out_sentence)):
                        s = StoT.sequence_to_text(out_sentence[n])
                        out_result_string.append(s)
                    #直接用out_result_string
                    target_sent = target.cpu().numpy().tolist()
                    for m in range((len(target_sent))):
                        t = StoT.sequence_to_text(target_sent[m])
                        tgt_result_string.append(t)
                    # tgt_result_string = list(map(StoT.sequence_to_text, target_sent))
                    embeddings_output = sentence_model.encode(out_result_string, convert_to_tensor=True)
                    embeddings_target = sentence_model.encode(tgt_result_string, convert_to_tensor=True)
                    cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
                    for i in range(len(target_sent)):
                        avg_cos += cos_sim[i][i]
                    avg_cos = avg_cos / len(target_sent)
                    eachSNR_avg_cos += avg_cos
                eachSNR_avg_cos = eachSNR_avg_cos / len(test_iterator)
                eachSNR_avg_cos.cpu().numpy()
                semantic_score.append(eachSNR_avg_cos)
            print("semantic score in 1 epoch:",semantic_score)


        #         output_sentences.append(result_string)
        #         target_sentences.append(target_sentence)
        #
        #         embeddings_output = sentence_model.encode(output_sentences, convert_to_tensor=True)
        #         embeddings_target = sentence_model.encode(target_sentences, convert_to_tensor=True)
        #         cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
        #         total_cos = 0
        #         for i in range(len(target_sentences)):
        #             print("sen1:{} \t\t sen2:{} \t\t Score: {:.4f}".format(target_sentences[i], output_sentences[i],
        #                                                                    cos_sim[i][i]))
        #             total_cos += cos_sim[i][i]
        #         los_cos = total_cos/len(target_sentences)
        # print(los_cos)

