# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/7/28 18:00”
A model used to get Source and Relay’s  information to help the destination to decode
"""
import torch
import numpy as np
import os
from  utils import SeqtoText, DENSE, Channel_With_PathLoss
