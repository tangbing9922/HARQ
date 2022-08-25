# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/7/15 16:02”
"""
import torch
import math
import numpy as np
import random

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

class Channels():
    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

class Channel2():
    # returns the message when passed through a channel.
    # AGWN, Fading
    # Note that we need to make sure that the colle map will not change in this
    # step, thus we should not use *= and +=.
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
        self.device = torch.device('cpu')

    def ideal_channel(self, _input):
        return _input

    def awgn(self, _input, _snr):
        _std = (10 ** (-_snr / 10.) / 2) ** 0.5
        _dim = _input.shape[0] * _input.shape[1] * _input.shape[2]
        spow = torch.sqrt(torch.sum(_input ** 2)) / _dim ** 0.5  # 信号功率？信号强度？
        _input = _input + torch.randn_like(_input) * _std * spow
        return _input

    def awgn_physical_layer(self, _input, _snr):
        _std = (10 ** (-_snr / 10.) / 2) ** 0.5
        _input = _input + torch.randn_like(_input) * _std
        return _input

    def phase_invariant_fading(self, _input, _snr):
        # ref from JSCC
        _dim = _input.shape[0] * _input.shape[1] * _input.shape[2]
        spow = torch.sqrt(torch.sum(_input ** 2)) / _dim ** 0.5
        _std = (10 ** (-_snr / 10.) / 2) ** 0.5 if self._iscomplex else (10 ** (-_snr / 10.)) ** 0.5
        _mul = (torch.randn(_input.shape[0], 1) ** 2 + torch.randn(_input.shape[0], 1) ** 2) ** 0.5
        _input = _input * _mul.view(-1, 1, 1).to(self.device)
        _input = _input + torch.randn_like(_input) * _std * spow
        return _input

    def phase_invariant_fading_physical_layer(self, _input, _snr):
        # ref from JSCC
        _std = (10 ** (-_snr / 10.) / 2) ** 0.5 if self._iscomplex else (10 ** (-_snr / 10.)) ** 0.5
        _mul = (torch.randn(_input.shape[0], 1) ** 2 + torch.randn(_input.shape[0], 1) ** 2) ** 0.5 # mul 是 信道增益吧
        _input = _input * _mul.view(-1, 1, 1).to(self.device)
        _input = _input + torch.randn_like(_input) * _std
        # print(_std)
        return _input


class Channel_With_PathLoss():
    def __init__(self):
        self.device = torch.device('cpu')

    def AWGN_Relay(self, Tx_sig, SNR, distance = 600):
        shape = Tx_sig.shape
        # dim = Tx_sig.shape[0] + Tx_sig.shape[1] + Tx_sig.shape[2]
        # spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        path_loss_exp = -2
        d_ref = 10
        PL = (distance / d_ref) ** path_loss_exp
        # Tx_sig = Tx_sig * PL
        std_no = (10 ** (- SNR / 10.) / 2) ** 0.5
        # Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no * spow
        Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no
        Tx_sig = Tx_sig.view(shape).to(self.device)
        return Tx_sig

    def AWGN_Direct(self, Tx_sig, SNR, distance = 1000):
        shape = Tx_sig.shape
        # dim = Tx_sig.shape[0] + Tx_sig.shape[1] + Tx_sig.shape[2]
        # spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        path_loss_exp = -3
        d_ref = 10
        PL = (distance / d_ref) ** path_loss_exp
        # Tx_sig = Tx_sig * PL
        std_no = (10 ** (- SNR / 10.) / 2) ** 0.5
        # Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no * spow
        Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no
        Tx_sig = Tx_sig.view(shape).to(self.device)
        return Tx_sig

    def Rayleigh_Relay(self, Tx_sig, SNR, distance = 600):
        shape = Tx_sig.shape
        dim = Tx_sig.shape[0] * Tx_sig.shape[1] * Tx_sig.shape[2]
        spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        path_loss_exp = -2
        d_ref  = 10
        PL = (distance / d_ref) ** path_loss_exp
        coe = ((torch.normal(0, PL**0.5, (Tx_sig.shape[0],1))**2 + torch.normal(0, PL**0.5, (Tx_sig.shape[0],1)) ** 2) ** 0.5) / (2 ** 0.5)
        std_no = (10 ** (- SNR / 10.) / 2) ** 0.5
        Tx_sig = Tx_sig * coe.view(-1, 1, 1).to(self.device)
        Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no * spow
        Tx_sig = Tx_sig.view(shape)
        return Tx_sig

    def Rayleigh_Direct(self, Tx_sig, SNR, distance = 1000):
        if distance == 1000:#或者范围
            path_loss_exp = -3
        shape = Tx_sig.shape
        dim = Tx_sig.shape[0] * Tx_sig.shape[1] * Tx_sig.shape[2]
        spow = torch.sqrt(torch.sum(Tx_sig ** 2)) / (dim ** 0.5)
        d_ref = 10
        PL = (distance / d_ref) ** path_loss_exp
        coe =  ((torch.normal(0, PL**0.5, (Tx_sig.shape[0],1))**2 + torch.normal(0, PL**0.5, (Tx_sig.shape[0],1))**2) ** 0.5) / (2 ** 0.5)
        std_no = (10 ** (- SNR / 10.) / 2) ** 0.5
        Tx_sig = Tx_sig * coe.view(-1, 1, 1).to(self.device)
        Tx_sig = Tx_sig + torch.randn_like(Tx_sig) * std_no * spow
        Tx_sig = Tx_sig.view(shape)
        return Tx_sig
        # 什么时候考虑 spow

# if __name__ == '__main__':
#     seed = 7
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     device = 'cpu'
#     torch.manual_seed(seed)  # set the seed for generating random numbers设置生成随机数的种子
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)
#     random.seed(seed)
#
#     Sig = torch.rand((16,30,128),dtype=torch.float)
#     print(Sig)
#     _iscomplex = True
#     SNR = 0
#     # std_noise = SNR_to_noise(SNR)
#     # channel1 = Channels()
#     # channel2 = Channel2(_iscomplex=_iscomplex)
#     # R_Sig = channel1.Rayleigh(Tx_sig = Sig, n_var = 0.717)
#     # R_sig2 = channel2.awgn_physical_layer(Sig, SNR)  # 感觉合理一点
#     channel = Channel_With_PathLoss()
#     R_sig = channel.AWGN_Direct(Sig, SNR)
#     # R_sig3 = channel.Rayleigh_Direct(Sig, SNR)
#     print(R_sig)
#     # print(R_sig2)