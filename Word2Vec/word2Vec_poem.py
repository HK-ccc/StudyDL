# 1. 从json文件获取数据，仅获取content内容，格式为list
# 2. 遍历content列表，用jieba的lcut函数进行切词
# 3. 在遍历过程中，将非标点符合和换行的切出来的词添加到结果中，再返回结果
import json
import jieba
import numpy as np
from tqdm import tqdm
import pickle

def CutWords(file_tang = "唐诗三百首.json", file_song = "宋词三百首"):
    stop_words = ["，", "。", "？", "\n"]
    result = []
    with open(file_tang, "r", encoding = "utf-8") as tang_file:
        tang = json.load(tang_file)
        tang_poem = [poem["content"] for poem in tang]
    for poem in tang_poem:
        cut_words = jieba.lcut(poem)
        result.append([word for word in cut_words if word not in stop_words])
    return result

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

if __name__ == "__main__":
    data = CutWords()
    word_2_index, index_2_word, word_2_onehot = GetDict(data)

    words_size = len(word_2_index)
    embedding_num = 108
    lr = 0.01
    epoch = 10  # 训练10轮
    n_gram = 3  # 预测前后3个词

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
                    # Embedding 生成嵌入向量
                    hidden = now_word_onehot @ w1
                    t = hidden @ w2
                    pre = softmax(t)
                    # 梯度计算和反向传播
                    G2 = pre - other_word_onehot # 误差
                    # delta_w2 是用于更新 w2 的梯度。根据误差 G2 和隐藏层的输出 hidden 计算 w2 的更新值
                    delta_w2 = hidden.T @ G2

                    # G1 是从输出层反传回隐藏层的误差，delta_w1 是用于更新 w1 的梯度。
                    # G1 通过 w2 的转置矩阵传播误差，结合中心词的 one-hot 向量，计算出 w1 的更新值
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1

                    # 更新权重
                    w1 -= lr * delta_w1
                    w2 -= lr * delta_w2
    
    with open("word2vec.pkl", "wb") as f:
        pickle.dump([w1, word_2_index, index_2_word, w2], f)

                    