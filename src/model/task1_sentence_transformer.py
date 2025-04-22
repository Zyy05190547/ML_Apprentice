# src/model/sentence_transformer.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class SentenceTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pooling="mean", output_dim=256):
        super(SentenceTransformer, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling.lower()
        self.hidden_size = self.backbone.config.hidden_size  #bert has embedding dim of 768 for each token
        self.output_dim = output_dim
        
        # MLP layer to project to output dimension
        self.projection = nn.Linear(self.hidden_size, self.output_dim)

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        use attention mask to perform avg pooling
        note: need this function since can't broadcast directly
        when multiply token_embeddings and attention_mask
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()) #(B*T*embedding_dim)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)          #(B*embedding_dim)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sentences):
        """
        Forward pass: use the second to last layers to get the feature map
        and then avg pool over each token to get the sentence embedding
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        
        output = self.backbone(**encoded_input)
        hidden_states = output.hidden_states
    
        token_embeddings = hidden_states[-2]
        pooled_embeddings = self.mean_pooling(token_embeddings, encoded_input['attention_mask'])
        
        # can change the output dim as warrent for diffenet architectures 
        # for later task specific heads
        projected_embeddings = self.projection(pooled_embeddings)
        
        return projected_embeddings


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentences = [
        "Hello world.",
        "How are you doing today?"
    ]

    #small output dim to showcase the result
    model_10 = SentenceTransformer(output_dim=10).to(device)
    embeddings_10 = model_10(sentences)
    print(f"sentence: '{sentences[0]}' --> embedding:{np.round(embeddings_10[0].detach().numpy(),4)}")
    print(f"sentence:'{sentences[1]}' --> embedding:{np.round(embeddings_10[1].detach().numpy(),4)}") 

    # Test 1: output_dim=128
    model_128 = SentenceTransformer(output_dim=128).to(device)
    embeddings_128 = model_128(sentences)
    print(f"Test 1 - Output dim 128: {embeddings_128.shape}")  # (2, 128)

    # Test 2: output_dim=256
    model_256 = SentenceTransformer(output_dim=256).to(device)
    embeddings_256 = model_256(sentences)
    print(f"Test 2 - Output dim 256: {embeddings_256.shape}")  # (2, 256)

    # Test 3: Single sentence
    model_single = SentenceTransformer(output_dim=64).to(device)
    embeddings_single = model_single(sentences[0])
    print(f"Test 3 - Single sentence Output dim 64: {embeddings_single.shape}")  # (1, 64)

    # Test 4: Empty input list
    try:
        model_empty = SentenceTransformer(output_dim=128).to(device)
        empty_input = []
        embeddings_empty = model_empty(empty_input)
        print(f"Test 4 - Empty input embeddings: {embeddings_empty.shape}")
    except Exception as e:
        print(f"Test 4 - error: {e}")

 

