import sentence_transformers
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
import torch


class SentenceSimilarity:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_similarity(self, sentence1, sentence2):
        embeddings1 = self.model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentence2, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores.item()

    def get_similarity_matrix(self, sentences1, sentences2):
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores.cpu().numpy()

    def get_most_similar(self, sentence, sentences):
        embeddings1 = self.model.encode(sentence, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return np.argmax(cosine_scores.cpu().numpy())

    def get_most_similar_sentence(self, sentence, sentences):
        return sentences[self.get_most_similar(sentence, sentences)]
    
    

if __name__ == '__main__':
    model_name = 'quora-distilbert-multilingual'
    sm = SentenceSimilarity(model_name)
    
    q = '你能给我讲解下蒙娜丽莎的微笑吗'
    a = '蒙娜丽莎'
    
    print(q, a,'相似度得分：',sm.get_similarity(q, a))