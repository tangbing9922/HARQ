'''
2021.12.21
'''
'''
将这个任务做成两个阶段的任务 
现在计算 句子丢进训练好的模型做相似度计算，以相似度计算作为损失
暂时不包含 Q_net
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
from utils import SeqtoText, greedy_decode
from preprocess_text import tokenize
from sentence_transformers import SentenceTransformer, util

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_file', default='europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint_path', default='/checkpoints/deepTest_cos&MI_32_1228/checkpoint_85.pth', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX_LENGTH', default=32, type=int)
parser.add_argument('--MIN_LENGTH', default=4, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_heads', default=8, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]#2

    # args.num_layers = 6
    # args.num_heads = 8
    # args.d_model = 256
    args.checkpoint_path = '..//model_channel_COS_MI/checkpoints/deepTest3_128_32_2022105/checkpoint_194.pth'
    # args.vocab_file = 'checkpoints/deepTest_32_1224/checkpoint_82.pth'#都只训练100个epoch，效果明显好过不加cos
    model = DeepTest(args.num_layers, num_vocab, num_vocab,
                        args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    # model.load_state_dict(torch.load("./checkpoints/deepTest_32_1222/checkpoint_50.pth"))
    model.load_state_dict(torch.load(args.checkpoint_path))
    #指定句子测试
    sentences = ['i am simply asking the commission to come and inform the committee on budgets before implementing the second tranche .','thank you very much. mr cox. i understand what you are saying. we have taken note of this.?','today is a sun day so i want to play ball with my friends']
    StoT = SeqtoText(token_to_idx, start_idx, end_idx)
    model.eval()
    with torch.no_grad():
        output_word = []
        target_word = []
        results = []
        listtest = []
        for sentence in sentences:
            tokens = [2 for _ in range(args.MAX_LENGTH)]  # pad 操作
            words = tokenize(sentence, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
            #print(words)
            #tokennize之后的单词列表
            tokens[0:len(words)] = [token_to_idx[word] for word in words]
            results.append(tokens)
            target = torch.tensor(results)
            target = target.to(device)
            #print(target)
            #索引列表
            ##########################################################################
            # 需要改
            out = greedy_decode(model, target, SNR, args.MAX_LENGTH, pad_idx, start_idx, args.channel)
            # print(out)
            out_sentences = out.cpu().numpy().tolist()
            t = np.array(out_sentences)
            print(t.shape)
            result_string = list(map(StoT.sequence_to_text, out_sentences))

        output_word = output_word + result_string
        target_word = sentences

        sentence_model = SentenceTransformer('models/sentence_model/training_stsbenchmark_continue_training-all-MiniLM-L6-v2-2021-11-25_20-55-16')

        embeddings_target = sentence_model.encode(target_word, convert_to_tensor=True)
        embeddings_output = sentence_model.encode(output_word, convert_to_tensor=True)

        cos_sim = util.pytorch_cos_sim(embeddings_target, embeddings_output)
        total_cos = 0
        print(cos_sim.shape)#torch.Size([3, 3])
        for i in range(len(target_word)):
            print("sen1:{} \t\t sen2:{} \t\t Score: {:.4f}".format(target_word[i], output_word[i], cos_sim[i][i]))
            total_cos += cos_sim[i][i]

        los_cos = total_cos/len(target_word)
        print(los_cos)
