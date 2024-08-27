#统计coco的词频在好几个network里都测一下，然后看一下在自己那个上面是不是keyword频率最高的。rank出来。citation network里找出来，就是print一些数据，出现频率的数据
#imagenet和coco，就不是完全不相交
#就是要找到subnetwork
#重点是retrieve，generation不需要操心
#试可以用bag of words，然后最后用一个fancy一些的，这样可控。
#dense就用open ai的ada的那个了
#统计词频和统计能不能retrieve出来subnetwork
#关键词放在profile里而不是summary，title abstract和keyword
#肯定有现成的methods或者包可以直接用来统计词频
#分类的groud truth就是现在的
#dense不好用都没事，dense就是不好用
#sparse的更好用
#先词频
#再看看bm25
#keyword匹配bm25可以搞定的

#做两张好看的图开会要讲
#model的部分我来讲
#Microsoft的流程图老赵说不行就白化了，花两个emb就行，画个fusion，过个gnn
#写论文有一些专门的网站，也有一些还有就是微软够用了，实验matlab python都有

'''
在ade20k的abstract中会有82次
'''
from collections import Counter
import re
import torch
import os

def tokenize(text):
    if not text:
        return ""
    """
    将文本拆分为单词，并去除标点符号和特殊字符。
    """
    # 移除标点符号和特殊字符，只保留字母和数字
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def build_vocab(corpus):
    """
    构建词汇表，包含语料库中所有独特的单词。
    """
    vocab = set()
    for text in corpus:
        tokens = tokenize(text)
        vocab.update(tokens)
    return sorted(vocab)

def bag_of_words(text, vocab):
    """
    将文本转换为词袋向量。
    """
    tokens = tokenize(text)
    word_counts = Counter(tokens)
    vector = [word_counts.get(word, 0) for word in vocab]
    return vector

def transform_corpus(corpus):
    """
    将整个语料库转换为词袋矩阵。
    """
    vocab = build_vocab(corpus)
    bag_of_words_matrix = [bag_of_words(text, vocab) for text in corpus]
    return vocab, bag_of_words_matrix

def bag_of_words_for_specific_vocab(text, specific_vocab):
    """
    将文本转换为特定词汇表的词袋向量。
    """
    tokens = tokenize(text)
    word_counts = Counter(tokens)
    vector = {word: word_counts.get(word, 0) for word in specific_vocab}
    return vector

def general_matrix(corpus):
    vocab, bow_matrix = transform_corpus(corpus)

    print("Vocabulary:", vocab)
    print("Bag of Words Matrix:")
    for vector in bow_matrix:
        print(vector)

def specific_matrix(corpus, specific_vocab):
    bow_vectors = [bag_of_words_for_specific_vocab(text, specific_vocab) for text in corpus]

    print("Specific Vocabulary:", specific_vocab)
    print("Bag of Words for Specific Vocabulary:")
    for vector in bow_vectors:
        print(vector)

def count_specific_words_in_corpus(corpus, specific_vocab):
    """
    计算特定词汇在整个语料库中出现的总次数。
    """
    total_counts = Counter()
    
    for text in corpus:
        tokens = tokenize(text)
        word_counts = Counter(tokens)
        # 只累加特定词汇的词频
        for word in specific_vocab:
            total_counts[word] += word_counts.get(word, 0)
    
    return total_counts[word]
if __name__ == "__main__":
    folder_path = '/home/ubuntu/Sci-Retriever-V2/dataset/'
    specific_vocab=['coco']
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        graph = torch.load(file_path)
        corpus = graph.content #先start with abstract，如果能跑通再用content
        total_count=count_specific_words_in_corpus(corpus, specific_vocab)
        print(f"Total Word Counts for Specific Vocabulary {specific_vocab} in paper {filename[:-3]}:")
        print(total_count,len(graph.abstract),total_count/len(graph.abstract))
