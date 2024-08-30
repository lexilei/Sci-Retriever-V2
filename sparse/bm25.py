#consider variations of bm25
import math
from collections import Counter
import torch

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_dl = sum(len(doc) for doc in documents) / len(documents)
        self.idf = self.compute_idf(documents)
        self.doc_freqs = [Counter(doc) for doc in documents]
        
    def compute_idf(self, documents):
        idf = {}
        num_docs = len(documents)
        
        for document in documents:
            for term in set(document):
                if term not in idf:
                    idf[term] = 0
                idf[term] += 1
        
        for term, freq in idf.items():
            idf[term] = math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))
        
        return idf
    
    def score(self, query, index):
        doc = self.documents[index]
        score = 0.0
        doc_length = len(doc)
        doc_counter = self.doc_freqs[index]
        
        for term in query:
            if term in doc_counter:
                tf = doc_counter[term]
                idf = self.idf.get(term, 0)
                score += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_dl)))
        
        return score
    
    def rank(self, query):
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

if __name__ == "__main__":
    graph = torch.load('/Users/lexlei/Sci-Retriever-V2/semantic-segmentation-on-ade20k.pt')

# 示例文档集合
    documents = [
        doc.lower().split() for doc in graph.abstract
    ]

    # 创建BM25实例
    bm25 = BM25(documents)

    # 查询
    query = "MASK dino".lower().split()

    # 排序文档根据查询
    ranked_docs = bm25.rank(query)

    # 输出结果
    for idx, score in ranked_docs:
        print(f"Document {idx}:  Score: {score}")
