import torch
import open_clip

"""
class OpenCLIPFeatureExtractor

Downloads the OpenCLIP model and provides methods to extract image and text features.

OpenCLIP models used: ViT-B-32, ViT-L-14
"""
class OpenCLIPFeatureExtractor:
    @torch.no_grad()
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess, self.tokenizer = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def get_text_features(self, text):
        """Extracts normalized text features."""
        text_tokens = self.tokenizer(text).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().tolist()

    @torch.no_grad()
    def get_image_features(self, image):
        """Extracts normalized image features."""
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().tolist()
    
    def truncate_text_by_tokens(self, text, max_tokens):
        """
        Truncates a given text to a specified number of tokens using OpenCLIP's tokenizer.
        
        Args:
            text (str): Input text to truncate.
            max_tokens (int): Maximum number of tokens to keep.

        Returns:
            str: Truncated text.
        """
        tokens = self.tokenizer(text)  # Tokenize the text
        tokens = tokens[0][:max_tokens]  # Truncate the token list
        
        # Ensure tokens are converted to a list of Python integers
        token_ids = [int(token.item()) for token in tokens]

        # Remove unknown/special tokens that may not be in the decoder dictionary
        token_ids = [t for t in token_ids if t in self.tokenizer.decoder]

        # Decode the tokens into text
        truncated_text = self.tokenizer.decode(token_ids)
        return truncated_text

