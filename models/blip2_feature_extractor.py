import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Blip2TextModelWithProjection, Blip2VisionModelWithProjection
import numpy as np

"""
class BLIP2FeatureExtractor

Loads the BLIP-2 model and provides methods to extract image and text features.
Projects both into a shared embedding space using projection models.

BLIP-2 model must be manually downloaded and loaded from Hugging Face.
"""
class BLIP2FeatureExtractor:
    @torch.no_grad()
    def __init__(self, text_model_name="Salesforce/blip2-itm-vit-g", vision_model_name="Salesforce/blip2-itm-vit-g"):
        print(f"Loading BLIP-2 Text Model: {text_model_name}")
        print(f"Loading BLIP-2 Vision Model: {vision_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = Blip2TextModelWithProjection.from_pretrained(text_model_name, torch_dtype=torch.float16).to(self.device)
        self.vision_model = Blip2VisionModelWithProjection.from_pretrained(vision_model_name, torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained(vision_model_name)

    @torch.no_grad()
    def get_text_features(self, text):
        """Extracts text embeddings from BLIP-2 text model with projection."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        text_outputs = self.text_model(**inputs)
        text_features = text_outputs.text_embeds  # Use projected text embeddings
        text_features /= torch.norm(text_features, dim=-1, keepdim=True)  # Normalize
        return [np.mean(text_features.cpu().squeeze(0).tolist(), axis = 0)]

    @torch.no_grad()
    def get_image_features(self, image):
        """Extracts image embeddings from BLIP-2 vision model with projection."""
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_outputs = self.vision_model(**inputs)
        image_features = image_outputs.image_embeds  # Use projected image embeddings
        image_features /= torch.norm(image_features, dim=-1, keepdim=True)
        return [np.mean(image_features.cpu().squeeze(0).tolist(), axis = 0)]
    
    def truncate_text_by_tokens(self, text, max_tokens):
        """
        Truncates a given text to a specified number of tokens using BLIP-2's tokenizer.
        
        Args:
            text (str): Input text to truncate.
            max_tokens (int): Maximum number of tokens to keep.

        Returns:
            str: Truncated text.
        """
        tokens = self.processor.tokenizer(text, truncation=False)["input_ids"]
        truncated_tokens = tokens[:max_tokens]  # Truncate to max_tokens
        truncated_text = self.processor.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text