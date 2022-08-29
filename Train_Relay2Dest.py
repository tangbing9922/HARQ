# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/8/25 9:48”

用于训练 中继R-> 信宿D 的模型
需要首先加载 从 信源S --> 中继R 的模型
值得一提的是，不能 加载 训好的 S --> R 的模型 来训练 R --> D的模型
因为S-->R模型加载之后不应该采用训练集数据作为输入

两种思路：
1.两部分模型单独训练，最后evaluate模式下进行整体的模型构建
2.两部分模型同时训练 从S->R->D一块整
思考上述两种方式的区别在哪儿？

参考Train_Destination_Without_Q.py
"""

def train(epoch, args, net1, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    _snr = torch.randint(4, 10,(1,))

    total = 0
    loss_record = []
    total_cos = 0
    total_MI = 0
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # mi = train_mi(net, mi_net, sents, noise_std[0], pad_idx, mi_opt, args.channel)
            mi = train_mi(net1, mi_net, sents, _snr, pad_idx, mi_opt, args.channel)
            loss, los_cos = semantic_block_train_step(net1, sents, sents, _snr, pad_idx, optimizer, criterion, args.channel, start_idx,
                                                      sentence_model, StoT, mi_net)
            # MI 和 semantic block 一块训练
            total += loss
            total_MI += mi
            loss_record.append(loss)
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            pbar.set_description(
                'Epoch:{};Type:Train; Loss: {:.3f}; MI{:.3f};los_cos:{:.3f}'.format(
                    epoch + 1, loss, mi, los_cos)
            )
        else:
            loss, los_cos = semantic_block_train_step(net1, sents, sents, _snr, pad_idx, optimizer, criterion, args.channel, start_idx,
                                                      sentence_model, StoT)
            total += loss
            los_cos = los_cos.cpu().detach().numpy()
            total_cos += los_cos
            total_cos = float(total_cos)
            loss_record.append(loss)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.3f}; los_cos: {:.3f}'.format(
                    epoch + 1, loss,los_cos
                )
            )
    return total / len(train_iterator), loss_record, total_cos / len(train_iterator), total_MI/ len(train_iterator)