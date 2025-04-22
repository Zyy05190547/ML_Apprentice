# src/model/sentence_transformer.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

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
    model = SentenceTransformer(output_dim=256)
    sentences = ["Hello world.", "How are you doing today?", "Sentence transformers are useful!"]
    embeddings = model(sentences)
    print("Embeddings shape:", embeddings.shape)  # Should be (3, 256)
    print("First sentence embedding (first 5 dims):", embeddings[0][:5])
