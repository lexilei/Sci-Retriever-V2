from openai import OpenAI
import pandas as pd
client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding



graph=torch.load('/home/ubuntu/Sci-Retriever-V2/semantic-segmentation-on-ade20k.pt')