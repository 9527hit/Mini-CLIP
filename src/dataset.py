import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

class CIFAR10Caption(CIFAR10):
    def __init__(self, root, train=True, download=True, tokenizer=None, max_length=32):
        # Define image augmentation/normalization
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # ResNet default input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(root, train=train, transform=transform, download=download)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # CIFAR-10 class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Simple Prompt Templates
        self.templates = [
            "a photo of a {}.",
            "this is a {}.",
            "a picture of the {}."
        ]

    def __getitem__(self, index):
        image, label_idx = super().__getitem__(index)
        label_name = self.classes[label_idx]
        
        # Construct text description
        # Randomly select a template for robustness
        import random
        template = random.choice(self.templates)
        caption = template.format(label_name)
        
        # Tokenize text
        encoded_text = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'label_idx': label_idx,
            'caption_text': caption
        }
