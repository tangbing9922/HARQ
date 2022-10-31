# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/10/31 10:22”
多跳 语义转发 模式 的性能分析 test 文件
"""
def multi_jump_test(model, num_jump:int, channel, args, SNR, StoT):
    model.eval()
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    score = []
    score1 = []
    test_data = EurDataset("test")
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                output_word = []
                target_word = []
                noise = SNR_to_noise(snr)
                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    trg_inp = target[:, :-1]
                    trg_real = target[:, 1:]
                    src_mask, look_ahead_mask = create_masks(target, trg_inp, pad_idx)

                    SD_channel = 'AWGN_Direct'
                    SR_channel = 'AWGN_Relay'
                    noise_std_SD = SNR_to_noise(0)
                    SD_output = greedy_decode(SD_model, target, noise_std_SD, args.MAX_LENGTH, pad_idx, start_idx, SD_channel)
                    SR_output = greedy_decode(SR_model, target, noise, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)
                    RD_output = greedy_decode(SR_model, SR_output, noise, args.MAX_LENGTH, pad_idx, start_idx, SR_channel)

                    SD_enc_output = SD_model.encoder(SD_output, src_mask)
                    SR_enc_output = SR_model.encoder(RD_output, src_mask)

                    out = greedy_decode4cross(model, target, SD_enc_output, SR_enc_output, args.MAX_LENGTH,
                                              pad_idx, start_idx)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    output_word = output_word + result_string
                    target_sent = target.cpu().numpy().tolist()
                    target_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + target_string
                Tx_word.append(target_word)
                Rx_word.append(output_word)
            bleu_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)
    score1 = np.mean(np.array(score), axis=0)
    print(score1)
    return score1

