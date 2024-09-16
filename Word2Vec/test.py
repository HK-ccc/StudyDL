# 训练模型
if __name__ == "__main__":
    # 分词并获取词频
    data, word_counts = CutWords()

    # 进行词频削峰并计算每个词的采样概率
    word_probs_dict = Subsampling(word_counts)

    # 获取词汇表，并将字典形式的概率转换为列表形式
    vocab = list(word_probs_dict.keys())
    word_probs_list = [word_probs_dict[word] for word in vocab]

    # 建立词典
    word_2_index, index_2_word, word_2_onehot = GetDict(data)

    vocab_size = len(word_2_index)
    embedding_num = 108
    lr = 0.01
    epoch = 10  # 训练10轮
    n_gram = 3  # 预测前后3个词
    num_neg_samples = 5  # 每个正样本采样5个负样本

    # 初始化两个权重矩阵，均使用正态分布（均值为 0，标准差为 1）进行初始化
    w1 = np.random.normal(0, 1, size=(vocab_size, embedding_num))
    w2 = np.random.normal(0, 1, size=(embedding_num, vocab_size))

    for e in range(epoch):
        for words in tqdm(data, desc=f"Epoch {e+1}/{epoch}"):
            for index, word in enumerate(words):
                if word not in word_2_index:
                    continue
                now_word_onehot = word_2_onehot[word]
                word_index = word_2_index[word]

                # 获取当前词的上下文词（前后n_gram个词）
                other_words = words[max(index - n_gram, 0):index] + words[index + 1: index + 1 + n_gram]

                # 对每个上下文词进行更新
                for other_word in other_words:
                    if other_word not in word_2_index:
                        continue
                    other_word_index = word_2_index[other_word]

                    # 正样本：中心词和上下文词对
                    hidden = now_word_onehot @ w1  # (1, embedding_num)
                    pos_score = hidden @ w2[:, other_word_index]  # (1,)
                    pos_sigmoid = 1 / (1 + np.exp(-pos_score))  # sigmoid(pos_score)
                    
                    # 计算正样本的梯度
                    G_pos = pos_sigmoid - 1  # (1,)
                    delta_w2_pos = hidden.T * G_pos  # (embedding_num, )
                    delta_w1_pos = now_word_onehot.T @ (G_pos * w2[:, other_word_index].reshape(-1, 1))  # (vocab_size, embedding_num)
                    
                    # 更新权重矩阵 w1 和 w2 对于正样本
                    w1 -= lr * delta_w1_pos
                    w2[:, other_word_index] -= lr * delta_w2_pos

                    # 负样本：根据词频采样负样本
                    neg_samples = NegativeSampling(word_probs_list, vocab, num_neg_samples=num_neg_samples)
                    for neg_sample_word in neg_samples:
                        neg_sample_index = word_2_index.get(neg_sample_word, None)
                        if neg_sample_index is None or neg_sample_index in [word_2_index[ow] for ow in other_words]:
                            continue  # 避免采样到正样本

                        neg_score = hidden @ w2[:, neg_sample_index]  # (1,)
                        neg_sigmoid = 1 / (1 + np.exp(-neg_score))  # sigmoid(neg_score)
                        
                        # 计算负样本的梯度
                        G_neg = neg_sigmoid  # (1,)
                        delta_w2_neg = hidden.T * G_neg  # (embedding_num, )
                        delta_w1_neg = now_word_onehot.T @ (G_neg * w2[:, neg_sample_index].reshape(-1, 1))  # (vocab_size, embedding_num)
                        
                        # 更新权重矩阵 w1 和 w2 对于负样本
                        w1 -= lr * delta_w1_neg
                        w2[:, neg_sample_index] -= lr * delta_w2_neg

    # 保存模型
    with open("word2vec_with_neg_sampling.pkl", "wb") as f:
        pickle.dump([w1, word_2_index, index_2_word, w2], f)
