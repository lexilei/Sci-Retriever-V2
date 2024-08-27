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

from collections import Counter
import re

def tokenize(text):
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

# 示例使用
corpus = [
    "I love programming in Python.",
    "Python programming is fun.",
    "I love coding!"
]

vocab, bow_matrix = transform_corpus(corpus)

print("Vocabulary:", vocab)
print("Bag of Words Matrix:")
for vector in bow_matrix:
    print(vector)
