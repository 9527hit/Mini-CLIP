import torch
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image
import torch.nn.functional as F
from src.model import MiniCLIP
import random

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and torch.backends.mps.is_built(): DEVICE = "mps"

def run_inference(image_path=None):
    # 1. Load Model
    print("Loading model...")
    model = MiniCLIP().to(DEVICE)
    try:
        model.load_state_dict(torch.load("mini_clip.pth", map_location=DEVICE))
    except FileNotFoundError:
        print("Error: mini_clip.pth not found. Please run train.py first.")
        return
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Define CIFAR-10 classes and construct sentences
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    prompts = [f"a photo of a {c}" for c in classes]
    
    # 3. Process Text
    text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    
    # 4. Get Image
    if image_path:
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    else:
        # Randomly from test set
        from torchvision.datasets import CIFAR10
        test_dataset = CIFAR10(root='./data', train=False, download=True)
        idx = random.randint(0, len(test_dataset)-1)
        image_raw, label_idx = test_dataset[idx]
        print(f"Ground Truth Label: {classes[label_idx]}")
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = preprocess(image_raw).unsqueeze(0).to(DEVICE)

    # 5. Inference
    with torch.no_grad():
        # Extract Features
        image_emb, text_emb = model(image_tensor, text_inputs['input_ids'], text_inputs['attention_mask'])
        
        # Calculate Similarity
        logits = (image_emb @ text_emb.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    # 6. Print Results
    print("\n--- Prediction Results ---")
    sorted_indices = probs.argsort()[::-1] # Descending
    for i in range(5): # Top 5
        idx = sorted_indices[i]
        print(f"{prompts[idx]:<25} : {probs[idx]*100:.2f}%")

if __name__ == "__main__":
    run_inference() 
