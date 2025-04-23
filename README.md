# ML Apprentice Project
This repository contains a multi-task learning model for sentence classification and sentiment analysis, built using a BERT-based architecture.

## Project Overview
The project consists of four main components:
1. **Sentence Transformer**: Custom sentence embedding generation using BERT
2. **Multi-Task Architecture**: A shared backbone with separate heads for genre classification and sentiment analysis
3. **Training Considerations**: Analysis of various fine-tuning strategies
4. **Training Loop Implementation**: A complete fine-tuning procedure with dummy data

## Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
## Task 1: Sentence Transformer
### Running the Code
```bash
python3 src/models/task1_sentence_transformer.py
```
In this part, I use a BERT model as the backbone and extract the feature map from the second-to-last layer. I chose an encoder-based network because I am interested in sentence classification (used later in Task 2), which requires each token to attend to future tokens in the input.

I avoided using the last layer as it may be task-specific and more compressed, potentially omitting general features. Likewise, I did not use early layers since they may not fully encode the sentence semantics.

After extracting the token-level embeddings, I applied average pooling to produce sentence-level embeddings. While average pooling is simple and effective, it may be lossy. One alternative I considered is to multiply the token vectors instead. However, this could lead to vanishing values due to vector norms shrinking. These methods could be evaluated during fine-tuning.


## Task 2: Multi-Task Architecture
A dual-headed architecture that performs both genre classification and sentiment analysis.
### Tasks
- **Task A**: Sentence classification across 5 literary genres
- **Task B**: Binary sentiment analysis (positive/negative)
### Architecture
- Shared BERT backbone for both tasks
- Task A Head: Deeper MLP with LayerNorm for genre classification
- Task B Head: Simpler MLP with LayerNorm for sentiment analysis
### Running the Code
```bash
cd src
python3 -m models.task2_multi_task_model
```
I used shared representation learning at the bottom and allowed task-specific specialization at the top. In future improvements, residual connections or additional normalization could be added to improve learning and training stability.


## Task 3: Training Considerations

In Task 3, I thought through a few possible ways to freeze or unfreeze parts of the network depending on the learning scenario. If the entire network is frozen, that means we are not fine-tuning any weights at all — we are simply using pretrained weights as-is, which is useful for fast inference but prevents the model from adapting to our tasks. If only the backbone is frozen but the task heads are still trainable, that allows us to keep the general structure and semantic knowledge from pretraining, while letting the model learn to adapt to task-specific outputs. If one of the task-specific heads is frozen, it allows the other task to keep learning without affecting a head that’s already well-performing — this is helpful in continual or staged training.

In a real transfer learning scenario, I would first select a pretrained model based on my task. For classification, I would choose an encoder-based model. For generation, I might go for a decoder or encoder-decoder. Then during training, I would freeze the pretrained layers for the first few epochs, just to let the new heads learn without interfering with the pretrained weights. Later, I would gradually unfreeze the encoder, usually starting from the top since it's closer to the output, so that the model can slowly adapt to the new domain without forgetting what it learned originally.



## Task 4: Training Loop
Implementation of the training procedure using synthetic data.
### Running the Code
```bash
cd ..
python3 src/train_task4.py
```

In this task, I used dummy data to simulate the fine-tuning of our multi-task model. Since the dataset is much smaller to the pretraining date, I kept the backbone frozen throughout training to avoid overfitting or wasting compute. I created a custom Dataset and DataLoader, and each batch includes both genre labels (for Task A) and sentiment labels (for Task B). For the loss function, I used CrossEntropyLoss for both tasks and introduced the option to assign weights to the two tasks — this allows us to prioritize one task over the other during training if desired.

Because multi-task loss values can be on different scales, I made sure to allow manual weighting in case we want to favor one task.  Currently, I’m printing training loss per epoch, but this could be improved by logging to TensorBoard or by adding validation metrics like accuracy or F1-score in future versions.
