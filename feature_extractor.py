"""
feature_extractor.py

This module provides a unified feature extractor supporting multiple models:
  - CLIP (Hugging Face)
  - ALIGN (Hugging Face)
  - OpenCLIP (open_clip library)
  - BLIP-2 (Hugging Face)
  - LongCLIP (Custom checkpoint)

Usage:
  extractor = FeatureExtractor(model_type="clip")
  text_features = extractor.get_text_features("Hello World")
  image_features = extractor.get_image_features(PIL_image)

Dependencies:
  - torch, numpy, and PIL for core operations.
  - Hugging Face Transformers for CLIP, ALIGN, and BLIP-2 models.
  - open_clip library for OpenCLIP.
  - Custom LongCLIP module for LongCLIP models.
"""
import torch
import numpy as np
from PIL import Image

# Hugging Face Transformers
from transformers import (
    AutoProcessor, AutoModel,
    Blip2TextModelWithProjection, Blip2VisionModelWithProjection
)

# OpenCLIP
import open_clip

# LongCLIP
from Long_CLIP.model import longclip
from Long_CLIP.model.simple_tokenizer import SimpleTokenizer


class FeatureExtractor:
    """
    A unified feature extractor supporting:
      - CLIP (Hugging Face)
      - ALIGN (Hugging Face)
      - OpenCLIP (open_clip library)
      - BLIP-2 (Hugging Face)
      - LongCLIP (Custom checkpoint)

    Usage:
      extractor = FeatureExtractor(model_type="clip")  # or "align", "openclip", "blip2", "longclip"
      text_features = extractor.get_text_features("Hello World")
      image_features = extractor.get_image_features(PIL_image)
    """

    @torch.no_grad()
    def __init__(self, model_type="clip", model_name=None, pretrained=None, checkpoint_path=None):
        """
        Initializes the selected model for feature extraction.

        Args:
            model_type (str): One of {"clip", "align", "openclip", "blip2", "longclip"}.
            model_name (str, optional): Model identifier (Hugging Face/OpenCLIP).
            pretrained (str, optional): Pretrained weights identifier (for OpenCLIP).
            checkpoint_path (str, optional): Required for LongCLIP (path to checkpoint file).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type.lower()

        if self.model_type in {"clip", "align"}:
            # CLIP and ALIGN (Hugging Face)
            if model_name is None:
                model_name = "openai/clip-vit-base-patch32" if self.model_type == "clip" else "kakaobrain/align-base"
            print(f"Loading {self.model_type.upper()} model: {model_name}")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)

        elif self.model_type == "openclip":
            # OpenCLIP (open_clip library)
            if model_name is None:
                model_name = "ViT-B-32"
            if pretrained is None:
                pretrained = "openai"
            print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
            self.model, self.preprocess, self.tokenizer = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

        elif self.model_type == "blip2":
            # BLIP-2 (Hugging Face)
            text_model_name = "Salesforce/blip2-itm-vit-g"
            vision_model_name = "Salesforce/blip2-itm-vit-g"
            print(f"Loading BLIP-2 Text Model: {text_model_name}")
            print(f"Loading BLIP-2 Vision Model: {vision_model_name}")
            self.text_model = Blip2TextModelWithProjection.from_pretrained(text_model_name, torch_dtype=torch.float16).to(self.device)
            self.vision_model = Blip2VisionModelWithProjection.from_pretrained(vision_model_name, torch_dtype=torch.float16).to(self.device)
            self.processor = AutoProcessor.from_pretrained(vision_model_name)

        elif self.model_type == "longclip":
            # LongCLIP (Custom checkpoint)
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for LongCLIP.")
            print(f"Loading LongCLIP model from {checkpoint_path}")
            self.model, self.preprocess = longclip.load(checkpoint_path, device=self.device)
            self.tokenizer = SimpleTokenizer()

        else:
            raise ValueError(f"Unknown model type '{model_type}'. Choose from 'clip', 'align', 'openclip', 'blip2', 'longclip'.")

    @torch.no_grad()
    def get_text_features(self, text):
        """
        Extracts normalized text features.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Normalized text feature vector.
        """
        if self.model_type in {"clip", "align"}:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**inputs)
        elif self.model_type == "openclip":
            text_tokens = self.tokenizer(text).to(self.device)
            text_features = self.model.encode_text(text_tokens)
        elif self.model_type == "blip2":
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            text_outputs = self.text_model(**inputs)
            text_features = text_outputs.text_embeds  # Use projected text embeddings
        elif self.model_type == "longclip":
            text_tokens = longclip.tokenize(text).to(self.device)
            text_features = self.model.encode_text(text_tokens)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if self.model_type == "blip2":
            text_features /= torch.norm(text_features, dim=-1, keepdim=True)
            return [np.mean(text_features.cpu().squeeze(0).tolist(), axis=0)]
        else:
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.tolist()

    @torch.no_grad()
    def get_image_features(self, image: Image.Image):
        """
        Extracts normalized image features.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            np.ndarray: Normalized image feature vector.
        """
        if self.model_type in {"clip", "align"}:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
        elif self.model_type in {"openclip", "longclip"}:
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)   
        elif self.model_type == "blip2":
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_outputs = self.vision_model(**inputs)
            image_features = image_outputs.image_embeds
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if self.model_type == "blip2":
            image_features /= torch.norm(image_features, dim=-1, keepdim=True)
            return [np.mean(image_features.cpu().squeeze(0).tolist(), axis = 0)]
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.tolist()

    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncates text to a specified number of tokens using the tokenizer 
        for the chosen model type.

        Args:
            text (str): The text to truncate.
            max_tokens (int): Maximum number of tokens to keep.

        Returns:
            str: The truncated text.
        """
        if self.model_type in {"clip", "align", "blip2"}:
            tokenizer = self.processor.tokenizer
            tokens = tokenizer(text, truncation=False)["input_ids"][:max_tokens]
            return tokenizer.decode(tokens, skip_special_tokens=True)

        elif self.model_type == "openclip":
            tokens = self.tokenizer(text)[0][:max_tokens]  # OpenCLIP returns a list of token tensors
            token_ids = [int(token.item()) for token in tokens]
            token_ids = [t for t in token_ids if t in self.tokenizer.decoder]  # Remove unknown tokens
            return self.tokenizer.decode(token_ids)

        elif self.model_type == "longclip":
            tokens = self.tokenizer.encode(text)[:max_tokens] # LongCLIP uses a custom tokenizer
            return self.tokenizer.decode(tokens)  # Decode back to text
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

