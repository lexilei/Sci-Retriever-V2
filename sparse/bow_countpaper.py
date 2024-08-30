from collections import Counter
import torch

def bow_count(papers, words):
    word_set = set(words)
    bow_list = []
    
    for paper in papers:
        paper_words = paper.lower().split()
        word_counts = Counter(paper_words)
        bow_dict = {word: word_counts[word] for word in word_set}
        bow_list.append(bow_dict)
    
    return bow_list

if __name__ == "__main__":
    graph = torch.load('/Users/lexlei/Sci-Retriever-V2/semantic-segmentation-on-ade20k.pt')

    # Example usage
    papers = graph.content

    words = ["dino"]

    bow_counts = bow_count(papers, words)
    for i, bow in enumerate(bow_counts):
        print(f"Paper {i + 1}: {bow}")
