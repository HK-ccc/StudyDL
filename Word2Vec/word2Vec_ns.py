# 1. 从json文件获取数据，仅获取content内容，格式为list
# 2. 遍历content列表，用jieba的lcut函数进行切词
# 3. 在遍历过程中，将非标点符合和换行的切出来的词添加到结果中，再返回结果
import json
import jieba
import numpy as np
from tqdm import tqdm
import pickle
import random
from collections import Counter

# 切词函数
# 从json文件获取content内容--如tang_poem--366首唐诗的诗句
# 遍历tang_poem，对每首诗进行切词处理，如果切分出来的词语不在停止词中，将其添加到result中
# 统计词频
def CutWords(file_tang = "唐诗三百首.json", file_song = "宋词三百首.json"):
    stop_words = ["，", "。", "？", "\n"]
    result = []
    word_freq = Counter() # 统计词频
    # 唐诗三百首
    with open(file_tang, "r", encoding = "utf-8") as tang_file:
        tang = json.load(tang_file)
        tang_poem = [poem["content"] for poem in tang]
    for poem in tang_poem:
        cut_words = jieba.lcut(poem)
        now_words = [word for word in cut_words if word not in stop_words]
        word_freq.update(now_words)
        result.append(now_words)
    # 宋词三百首
    with open(file_song, "r", encoding = "utf-8") as song_file:
        song = json.load(song_file)
        song_poem = [poem["content"] for poem in song]
    for poem in song_poem:
        cut_words = jieba.lcut(poem)
        now_words = [word for word in cut_words if word not in stop_words]
        word_freq.update(now_words)
        result.append(now_words)

    total_count = sum(word_freq.values())
    # 将词频转为词概率，并进行3/4削峰
    word_freq = {word: (count / total_count)**0.75 for word, count in word_freq.items()}
    total_freq = sum(word_freq.values())
    #  归一化为概率分布
    word_freq = {word: freq / total_freq for word, freq in word_freq.items()}
    return result, word_freq

# 4. 建立训练需要用到的变量，分别是word_2_index(字典), index_2_word(列表), word_2_onehot(列表)
#    示例格式：word_2_index = {"西陆" : 0, "白发" : 1}
#             index_2_word = ["西陆", "白发"]
#             word_2_onehot = {"西陆":[1,0,0,……]}
# 5. 

def GetDict(data):
    index_2_word = []
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)
    word_2_index = {word:index for index,word in enumerate(index_2_word)}
    words_size = len(word_2_index)
    word_2_onehot = {}
    for word, index in word_2_index.items():
        one_hot = np.zeros((1, words_size))
        one_hot[0, index] = 1
        word_2_onehot[word] = one_hot
    return word_2_index, index_2_word, word_2_onehot

def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis = 1, keepdims = True)

# 定义负采样函数，根据词频采样负样本，
# @voc_table--基于词频的采样分布
# @num_neg_samples--负采样样本数量
# @word_index 正样本词的索引
# @excluded_words 正样本上下文词的索引
# @return 列表，元素是负样本的索引集合
def NegativeSampling(voc_table, num_neg_samples, word_index, excluded_words):
    neg_samples = []
    while len(neg_samples) < num_neg_samples:
        sample = np.random.choice(len(voc_table), p=voc_table)  # 根据词频分布采样
        if sample != word_index and sample not in excluded_words:
            neg_samples.append(sample)
    return neg_samples

if __name__ == "__main__":
    data, word_freq = CutWords()
    word_2_index, index_2_word, word_2_onehot = GetDict(data)

    words_size = len(word_2_index)
    embedding_num = 108
    lr = 0.01
    epoch = 10  # 训练10轮
    n_gram = 3  # 预测前后3个词
    num_neg_samples = 5 # 每个正样本采样5个负样本

    # 初始化两个权重矩阵，均使用正态分布（均值为 0，标准差为 1）进行初始化
    w1 = np.random.normal(0,1,size = (words_size, embedding_num))
    w2 = np.random.normal(0,1,size = (embedding_num, words_size))

    for e in range(epoch):  
        for words in tqdm(data):
            for index, word in enumerate(words):
                # 获取当前词语的onehot向量
                now_word_onehot = word_2_onehot[word]
                # 获取当前词语的前后3个词语，即滑动窗口大小为7
                other_words = words[max(index - n_gram, 0):index] + words[index+1: index+1+n_gram]
                # 计算预测损失
                for other_word in other_words:
                    other_word_onehot = word_2_onehot[other_word]
                    # 计算正样本的损失，更新权重w1和w2
                    hidden = now_word_onehot @ w1
                    t = hidden @ w2
                    pre = softmax(t)
                    G2 = pre - other_word_onehot
                    delta_w2 = hidden.T @ G2
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1
                    w1 -= lr * delta_w1
                    w2 -= lr * delta_w2

                    # 负样本：根据词频采样负样本
                    neg_samples = NegativeSampling(word_freq, num_neg_samples, word_2_index[other_word], [word_2_index[n_word] for n_word in other_words] + [index])
                    for neg_sample_word in neg_samples:
                        neg_sample_index = word_2_index[neg_sample_word]

                        neg_score = hidden @ w2[:, neg_sample_index]  # (1,)
                        neg_sigmoid = 1 / (1 + np.exp(-neg_score))  # sigmoid(neg_score)
                        
                        # 计算负样本的梯度
                        G_neg = neg_sigmoid  # (1,)
                        delta_w2_neg = hidden.T * G_neg  # (embedding_num, )
                        delta_w1_neg = now_word_onehot.T @ (G_neg * w2[:, neg_sample_index].reshape(-1, 1))  # (vocab_size, embedding_num)
                        
                        # 更新权重矩阵 w1 和 w2 对于负样本
                        w1 -= lr * delta_w1_neg
                        w2[:, neg_sample_index] -= lr * delta_w2_neg
    
    with open("word2vec.pkl", "wb") as f:
        pickle.dump([w1, word_2_index, index_2_word, w2], f)
