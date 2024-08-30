import torch
graph = torch.load('/Users/lexlei/Sci-Retriever-V2/semantic-segmentation-on-ade20k.pt')
print(graph.title[195])
print(graph.abstract[195])
print(graph.content[195])