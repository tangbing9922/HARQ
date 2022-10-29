# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/29 14:46”
训练 cross_attention 模块 注意 直接链路 和 中继链路  都不解码
特征->信道 ->cross attention
"""
import torch

def getFeature_afterChannel(model, src, noise_std, max_len, padding_idx, start_symbol, channel):
    model.eval()
    # create src_mask
    channels = Channel_With_PathLoss()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    if channel == 'AWGN_Relay':
        Rx_sig = channels.AWGN_Relay(Tx_sig, noise_std)
    elif channel == 'AWGN_Direct':
        Rx_sig = channels.AWGN_Direct(Tx_sig, noise_std)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")
    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    # torch.tensor.fill_(x)用指定的值x填充张量
    # torch.tensor.type_as(type) 将tensor的类型转换为给定张量的类型
    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.predict(dec_output)

        # predict the output_sentences
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs