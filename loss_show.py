# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/26 20:35”
"""
from torch.utils.tensorboard import SummaryWriter
import time
import random

writer = SummaryWriter(log_dir='./' + time.strftime('%y-%m-%d_%H.%M', time.localtime())) # 确定路径
for i in range(10):
    # 直接写入就行，更多用法请参考后面的链接，这里就是最最最简单的例子
    writer.add_scalar('test_value', random.random(), i)
# 想写的都写完后调用这个函数
writer.close()