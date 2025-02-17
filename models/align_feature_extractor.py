import torch
from transformers import AutoProcessor, AutoModel

"""
class ALIGNFeatureExtractor

Downloads the ALIGN model from Hugging Face and provides methods to extract image and text features.

ALIGN model used: kakaobrain/align-base
"""
class ALIGNFeatureExtractor:
    @torch.no_grad()
    def __init__(self, model_name="kakaobrain/align-base"):
        print(f"Loading ALIGN model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def get_text_features(self, text):
        """Extracts normalized text features."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().tolist()

    @torch.no_grad()
    def get_image_features(self, image):
        """Extracts normalized image features."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().tolist()
    
    def truncate_text_by_tokens(self, text, max_tokens):
        """
        Truncates a given text to a specified number of tokens using the Hugging Face ALIGN tokenizer.
        
        Args:
            text (str): Input text to truncate.
            max_tokens (int): Maximum number of tokens to keep.

        Returns:
            str: Truncated text.
        """
        tokenizer = self.processor.tokenizer
        tokens = tokenizer(text, truncation=False)["input_ids"][:max_tokens]
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text
