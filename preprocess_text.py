# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: text_preprocess.py
@Time: 2021/3/31 22:14
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:44:08 2020

@author: hx301
"""
import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_dir', default='/europarl/en', type=str)
#1130
# parser.add_argument('--input_data_dir', default='/europarl/1130', type=str)
parser.add_argument('--output_train_dir', default='/europarl/train_data50.pkl', type=str)
parser.add_argument('--output_test_dir', default='/europarl/test_data50.pkl', type=str)
parser.add_argument('--output_vocab', default='/europarl/vocab50.json', type=str)
#behind is 11.4 new added
parser.add_argument('--output_col_sen', default='/europarl/colofsen50.txt', type=str)

SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}
#由于所用数据集当中txt文档均采用utf-8(符合unicode标准)编码
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
#'Mn'类别为mark nonspacing（Nonspacing Mark）
#a nonspacing combining mark (zero advance width) (谷歌翻译: 一个非空格组合标志（零超前宽度）)。
#简而言之就是丢弃这种稀奇古怪的字符
#python官方文档进行unicodedata模块学习
#unicodedata.normalize(form,unistr) 
#返回 Unicode 字符串 unistr 的正常形式 form 。 
#form 的有效值为 ‘NFC’ 、 ‘NFKC’ 、 ‘NFD’ 和 ‘NFKD’ 。
#unicodedata.category(chr)
#返回分配给字符 chr 的常规类别为字符串。

def normalize_string(s):
    # normalize unicode characters标准化unicode字符
    s = unicode_to_ascii(s)
    # remove the XML-tags字面意思
    s = remove_tags(s)
    '''
    from w3lib.html import remove_tags
    a = '<em><em>ai</em></em>工程师'
    print(remove_tags(a))
    #ai工程师
    '''
    # add white space before !.?
    #替换
    s = re.sub(r'([!.?])', r' \1', s)#\1是匹配第一个
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)#匹配任意空白符，次数为一次或多次
    # change to lower letter，所有字符变为小写。
    s = s.lower()
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=50):
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH-1:#将最大长度设置为 max_length-1，这样在加了start 和 end 之后长度仍为max_len
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines

def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)

def process(text_path):
    fop = open(text_path, 'r', encoding='utf8')
    raw_data = fop.read()
    sentences = raw_data.strip().split('\n')
    raw_data_input = [normalize_string(data) for data in sentences]
    raw_data_input = cutted_data(raw_data_input)
    fop.close()

    return raw_data_input


def tokenize(s, delim=' ',  add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, token_to_idx = { }, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, ):
    token_to_count = {}

    for seq in sequences:
      seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                      punct_to_remove=punct_to_remove,
                      add_start_token=False, add_end_token=False)
      for token in seq_tokens:
        if token not in token_to_count:
          token_to_count[token] = 0
        token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
      if count >= min_token_count:
        token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
      if token not in token_to_idx:
        if allow_unk:
          token = '<UNK>'
        else:
          raise KeyError('Token "%s" not in vocab' % token)
      seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
      tokens.append(idx_to_token[idx])
      if stop_at_end and tokens[-1] == '<END>':
        break
    if delim is None:
      return tokens
    else:
      return delim.join(tokens)


def main(args):
    data_dir = 'E:/Desktop/tb/coding/code/DeepSC-master'
    args.input_data_dir = data_dir + args.input_data_dir
    args.output_train_dir = data_dir + args.output_train_dir
    args.output_test_dir = data_dir + args.output_test_dir
    args.output_vocab = data_dir + args.output_vocab
    #behind 1 line is  11.4 new added
    args.output_col_sen = data_dir + args.output_col_sen
    print(args.input_data_dir)
    sentences = []
    print('Preprocess Raw Text')
    for fn in tqdm(os.listdir(args.input_data_dir)):
        if not fn.endswith('.txt'): continue
        process_sentences = process(os.path.join(args.input_data_dir, fn))
        sentences += process_sentences

    # remove the same sentences
    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1
    sentences = list(a.keys())
    print('Number of sentences: {}'.format(len(sentences)))
    
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )

    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))

    # save the vocab
    if args.output_vocab != '':
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)

    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)


    print('Writing Data')
    train_data = results[: round(len(results) * 0.9)]
    #round函数
    test_data = results[round(len(results) * 0.9):]
    # 学习将处理过后的sentences保存为.txt
    with open(args.output_col_sen, 'w') as f1:
        f1.write(str(sentences))
    with open("E:/Desktop/tb/coding/code/DeepSC-master/europarl/1130.txt",'w') as f2:
        f2.write(str(train_data))
    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)
    # f1 = open('D:/Users/Administrator/Desktop/DeepSC-master/europarl/train_data.pkl')  
    # info = pickle.load(f)
    # print(info)
if __name__ == '__main__':
    args = parser.parse_args()
    #parser.parse_args()返回的是带有成员的Namespace
    #namespace - 用于获取属性的对象。 默认值是一个新的空 Namespace 对象
    '''
    class argparse.Namespace
由 parse_args() 默认使用的简单类，可创建一个存放属性的对象并将其返回。

这个类被有意做得很简单，只是一个具有可读字符串表示形式的 object。
    '''
    main(args)