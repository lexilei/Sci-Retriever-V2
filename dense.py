'''
the following file utilizes specter to generate
total paper in semantic segmentation on ade20k is 763
testing with batch size 50 is aout max, 80 exceeds limit
'''
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm
import torch

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

#load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
#other possibilities: allenai/specter2_<classification|regression|adhoc_query>
graph=torch.load('/home/ubuntu/Sci-Retriever-V2/semantic-segmentation-on-ade20k.pt')

papers=[]
for i in range(50):
    papers.append({'title':graph.title[i],'abstract':graph.abstract[i]})

# concatenate title and abstract
text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]

inputs = tokenizer(text_batch, padding=True, truncation=True,return_tensors="pt", return_token_type_ids=False, max_length=512)
output = model(**inputs)
embeddings = output.last_hidden_state[:, 0, :]
torch.save(embeddings, "/home/ubuntu/Sci-Retriever-V2/specter2_generated.pt")