from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.metrics.pairwise import euclidean_distances
import torch

def embed_input(text_batch):
  # preprocess the input
  inputs = tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
  output = model(**inputs)
  # take the first token in the batch as the embedding
  embeddings = output.last_hidden_state[:, 0, :]
  return embeddings

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

#load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

#load the query adapter, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="specter2_adhoc_query", set_active=True)
query = ["We introduce a new language representation model called BERT"]
query_embedding = embed_input(query)

#load the proximity adapter, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="specter2_proximity", set_active=True)
papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]
# concatenate title and abstract
text_papers_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
paper_embeddings = embed_input(text_papers_batch)
# paper_embeddings=torch.load('/home/ubuntu/Sci-Retriever-V2/specter2_generated.pt')
# print(paper_embeddings.detach().numpy())
#Calculate L2 distance between query and papers
l2_distance = euclidean_distances(paper_embeddings.detach().numpy(), query_embedding.detach().numpy()).flatten()
print(l2_distance)