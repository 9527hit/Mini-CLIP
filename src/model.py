import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertConfig

class MiniCLIP(nn.Module):
    def __init__(self, embed_dim=256, freeze_backbone=False):
        super(MiniCLIP, self).__init__()
        
        # --- Image Encoder (ResNet18) ---
        # Use pre-trained weights
        self.image_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove original classification head
        self.image_encoder.fc = nn.Identity() 
        
        # --- Text Encoder (DistilBERT) ---
        # Use HuggingFace pre-trained model
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # --- Projection Heads ---
        # ResNet18 output is 512 dim
        self.image_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
        # DistilBERT output is 768 dim
        self.text_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, embed_dim)
        )
        
        # Logit Scale (Temperature), initial value log(1/0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, image, text_input_ids, text_attention_mask):
        # 1. Image Encoding
        image_features = self.image_encoder(image) # [batch, 512]
        image_embeddings = self.image_projection(image_features) # [batch, embed_dim]
        
        # 2. Text Encoding
        text_output = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        # Use [CLS] token (0th token) for sentence feature
        text_features = text_output.last_hidden_state[:, 0, :] # [batch, 768]
        text_embeddings = self.text_projection(text_features) # [batch, embed_dim]
        
        # 3. L2 Normalization
        # Critical step: ensure vectors are on the unit hypersphere
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        
        return image_embeddings, text_embeddings
