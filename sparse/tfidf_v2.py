'''
high tfidf if high uniqueness
所以可以用tfidf来扒文中的关键词

文档长度会有影响，可以应用这个technique来减少文档长度的影响：
最大词频归一化:

将每个词的TF除以该文档中出现次数最多的词的频率。
公式为：TF(word, doc) = 0.5 + 0.5 * (count(word, doc) / max_count(doc))
这种方式可以确保最频繁的词TF值为1，而其他词的TF值相对较小，文档长度对TF的影响被削弱。

或者用bm25的公式
'''
import math
from collections import Counter
import re
import torch

def tokenize(text):
    """
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def term_frequency(text):
    """
    计算文本中每个词的词频（TF）。
    """
    tokens = tokenize(text)
    word_counts = Counter(tokens)
    total_words = len(tokens)
    tf = {word: count / total_words for word, count in word_counts.items()}
    return tf

def inverse_document_frequency(corpus):
    """
    计算语料库中每个词的逆文档频率（IDF）。
    """
    num_docs = len(corpus)
    word_doc_count = Counter()
    
    for text in corpus:
        tokens = set(tokenize(text))  # 只统计词在文档中是否出现，不考虑重复
        for token in tokens:
            word_doc_count[token] += 1
    
    idf = {word: math.log(num_docs / (count)) for word, count in word_doc_count.items()}
    return idf

def tf_idf(corpus):
    """
    计算整个语料库的TF-IDF值。
    """
    idf = inverse_document_frequency(corpus)
    tf_idf_scores = []
    
    for text in corpus:
        tf = term_frequency(text)
        tf_idf = {word: tf_val * idf[word] for word, tf_val in tf.items()}
        tf_idf_scores.append(tf_idf)
    
    return tf_idf_scores

if __name__ == "__main__":
    graph = torch.load('/Users/lexlei/Sci-Retriever-V2/semantic-segmentation-on-ade20k.pt')
# 示例使用
    corpus = graph.content

    # 计算语料库的TF-IDF值
    tf_idf_scores = tf_idf(corpus)

    # 打印TF-IDF结果
    for i, doc_scores in enumerate(tf_idf_scores):
        print(f"Document {i+1} TF-IDF Scores:")
        for word, score in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True):
            print(f"  {word}: {score:.4f}")
