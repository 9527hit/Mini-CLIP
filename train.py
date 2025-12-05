import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import os

from src.model import MiniCLIP
from src.dataset import CIFAR10Caption

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps" # Mac support
BATCH_SIZE = 64 
EPOCHS = 5
LEARNING_RATE = 1e-4

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 2. Prepare Data
    dataset = CIFAR10Caption(root='./data', train=True, download=True, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 3. Initialize Model
    model = MiniCLIP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        
        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            img_emb, text_emb = model(images, input_ids, attention_mask)
            
            # Calculate Similarity Logits
            # (Batch, Batch)
            logits = (img_emb @ text_emb.T) * model.logit_scale.exp()
            
            # Construct Labels: [0, 1, 2, ..., Batch-1]
            labels = torch.arange(images.shape[0]).to(DEVICE)
            
            # Symmetric Cross Entropy Loss
            loss_i = nn.CrossEntropyLoss()(logits, labels)      # Image -> Text
            loss_t = nn.CrossEntropyLoss()(logits.T, labels)    # Text -> Image
            loss = (loss_i + loss_t) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # Save Model
        torch.save(model.state_dict(), "mini_clip.pth")
        print("Model saved to mini_clip.pth")

if __name__ == "__main__":
    train()
