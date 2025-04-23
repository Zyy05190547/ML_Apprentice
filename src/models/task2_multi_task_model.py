import torch
import torch.nn as nn
from .task1_sentence_transformer import SentenceTransformer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling="mean",
        projection_dim=1024,   # 'upsampling' after the transformer
        hidden_dims_task_a=[512, 256, 128],
        hidden_dims_task_b=[512,128],
        num_classes_task_a=5,  # 5 literature genres
        num_classes_task_b=2,  # positive/negative
    ):
        super(MultiTaskSentenceTransformer, self).__init__()
        
        # Shared encoder
        self.encoder = SentenceTransformer(model_name=model_name, pooling=pooling, output_dim=projection_dim)
        
        # Task A Head for genre classification
        self.task_a_head = self._build_mlp_head(
            input_dim=projection_dim,
            hidden_dims=hidden_dims_task_a,
            output_dim=num_classes_task_a
        )
        
        # Task B  Head for sentiment analysis
        self.task_b_head = self._build_mlp_head(
            input_dim=projection_dim,
            hidden_dims=hidden_dims_task_b,
            output_dim=num_classes_task_b
        )

    def _build_mlp_head(self, input_dim, hidden_dims, output_dim):
        """
        Helper function to build an MLP head using nn.Sequential.
        """
        layers = [nn.LayerNorm(input_dim)]
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))  # Final output layer
        return nn.Sequential(*layers)

    def forward(self, sentences):
        """
        Forward pass:
        - Encode sentences
        - Predict both tasks
        """
        embeddings = self.encoder(sentences)  # shape (batch_size, projection_dim)
        
        task_a_logits = self.task_a_head(embeddings)  # (batch_size, 5)
        task_b_logits = self.task_b_head(embeddings)  # (batch_size, 2)
        
        return {
            "task_a_logits": task_a_logits,
            "task_b_logits": task_b_logits
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentences = [
        "Thou art more lovely and more temperate.",
        "All the world’s a stage, and all the men and women merely players.",
        "The lady doth protest too much, methinks.",
        "Cowards die many times before their deaths; the valiant never taste of death but once.",
        "Good night, good night! Parting is such sweet sorrow."
    ]

    model = MultiTaskSentenceTransformer(
        model_name="bert-base-uncased",
        pooling="mean",
        projection_dim=1024,  #'upsampling' after the transformer
        hidden_dims_task_a=[512, 256, 128],
        hidden_dims_task_b=[512,128],
        num_classes_task_a=5, # 5 made up genres
        num_classes_task_b=2  # 2 sentiment classes: positive/negative
    ).to(device)

    outputs = model(sentences)

    print(outputs)
    print(f"Task A logits shape (genres): {outputs['task_a_logits'].shape}")
    print(f"Task B logits shape (sentiment): {outputs['task_b_logits'].shape}")
