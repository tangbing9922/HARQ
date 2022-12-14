# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/12/6 19:22”
加载DeepTest 微调 quant_deepsc
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, Quant_train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import QUANT_DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='./europarl/vocab32.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/Train_SemanticBlock_Rayleigh_Relay', type=str)
parser.add_argument('--quant-checkpoint-path', default='checkpoints/Quantization_4bit_Rayleigh_Relay', type=str)
parser.add_argument('--channel', default='Rayleigh_Relay', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=32, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    # noise = SNR_to_noise(0)
    dequant_after_channel = True
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, noise_std[0], pad_idx,
                            criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total / len(test_iterator)

def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    # dequant_after_channel = True
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(18), size=(1))
    # noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(30), size=(1))
    noise_std = noise_std.astype(np.float64)[0]
    # noise = SNR_to_noise(0)
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, noise_std, pad_idx, mi_opt, args.channel)
            loss = Quant_train_step(net, sents, sents, noise_std, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = Quant_train_step(net, sents, sents, noise_std, pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

if __name__ == '__main__':
    setup_seed(7)
    args = parser.parse_args()
    # args.vocab_file = '/data/' + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = QUANT_DeepSC(args.num_layers, num_vocab, num_vocab,
                          num_vocab, num_vocab, args.d_model, args.num_heads,
                          args.dff, 0.1, num_bit=4).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=2e-5)
    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)

    is_quant = True
    if is_quant:  # 加载预训练模型
        pretrained_dict = torch.load(args.checkpoint_path + '/1129DeepTest_net_checkpoint.pth')
        deepsc_dict = deepsc.state_dict()

        new_dict = {k: v for k, v in pretrained_dict.items() if k in deepsc_dict.keys()}
        deepsc_dict.update(new_dict)
        deepsc.load_state_dict(deepsc_dict)
        save_path = args.quant_checkpoint_path
    else:
        save_path = args.checkpoint_path

    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, deepsc, mi_net)  # 训练
        avg_acc = validate(epoch, args, deepsc)  # 验证

        if avg_acc < record_acc:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []