import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model.task2_multi_task_model import MultiTaskSentenceTransformer

# Dummy Multi-Task Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, sentences, task_a_labels, task_b_labels):
        self.sentences = sentences
        self.task_a_labels = task_a_labels
        self.task_b_labels = task_b_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "task_a_label": self.task_a_labels[idx],
            "task_b_label": self.task_b_labels[idx]
        }

# dummy data
sentences = [
        "Thou art more lovely and more temperate.",
        "All the world’s a stage, and all the men and women merely players.",
        "The lady doth protest too much, methinks.",
        "Cowards die many times before their deaths; the valiant never taste of death but once.",
        "Good night, good night! Parting is such sweet sorrow."
    ]

task_a_labels = torch.tensor([0, 1, 2, 3, 4])  # 5 genre classes
task_b_labels = torch.tensor([1, 0, 0, 0, 1])  # 2 sentiment classes

# Build dataset and dataloader
dataset = MultiTaskDataset(sentences, task_a_labels, task_b_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskSentenceTransformer(
    model_name="bert-base-uncased",
    pooling="mean",
    projection_dim=256,
    hidden_dims_task_a=[512, 256, 128],  # deeper for genre
    hidden_dims_task_b=[128],            # shallower for sentiment
    num_classes_task_a=5,
    num_classes_task_b=2
).to(device)

# leave the weights of the transformer encoder frozen
# and only train the task-specific heads
for param in model.encoder.backbone.parameters():
    param.requires_grad = False

loss_a_weight = 1
loss_b_weight = 1
loss_fn_task_a = nn.CrossEntropyLoss()
loss_fn_task_b = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# main training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        sentences_batch = batch["sentence"]
        labels_a = batch["task_a_label"].to(device)
        labels_b = batch["task_b_label"].to(device)

        optimizer.zero_grad()

        outputs = model(sentences_batch)
        logits_a = outputs["task_a_logits"]
        logits_b = outputs["task_b_logits"]

        loss_a = loss_fn_task_a(logits_a, labels_a)
        loss_b = loss_fn_task_b(logits_b, labels_b)
        #could weigh these differently if want a focus only one over the other
        #also need to make sure they are on the same scale
        loss = loss_a_weight*loss_a + loss_b_weight*loss_b 

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    pass
