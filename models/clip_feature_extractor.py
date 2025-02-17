import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

"""
class CLIPFeatureExtractor

Downloads CLIP model and has methods to embed images and images into a shared vector space

CLIP models, used: clip-vit-large-patch14, clip-vit-base-patch32 
"""
class CLIPFeatureExtractor:
    @torch.no_grad()
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def get_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = inputs.to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.tolist()
        return text_features

    @torch.no_grad()
    def get_image_features(self, images):
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.tolist()
        
        return image_features
    
    def truncate_text_by_tokens(self, text, max_tokens):
        
        tokens = self.tokenizer(text, truncation=False)["input_ids"]
        truncated_tokens = tokens[:max_tokens]  
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text

