import torch
from Long_CLIP.model import longclip
from Long_CLIP.model.simple_tokenizer import SimpleTokenizer


"""
class LongCLIPFeatureExtractor

Loads the Long-CLIP model and provides methods to extract image and text features.

Long-CLIP model must be manually downloaded and loaded from checkpoints.
"""
class LongCLIPFeatureExtractor:
    @torch.no_grad()
    def __init__(self, checkpoint_path: str ="Long_CLIP/checkpoints/longclip-B.pt"):
        print(f"Loading Long-CLIP model from {checkpoint_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = longclip.load(checkpoint_path, device=self.device)
        self.tokenizer = SimpleTokenizer()

    @torch.no_grad()
    def get_text_features(self, text_list):
        """Extracts normalized text features from Long-CLIP."""
        text_tokens = longclip.tokenize(text_list).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    @torch.no_grad()
    def get_image_features(self, image):
        """Extracts normalized image features from Long-CLIP.""" 
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def truncate_text_by_tokens(self, text, max_tokens):
        """
        Truncates a given text to a specified number of tokens using Long-CLIP's tokenizer.
        
        Args:
            text (str): Input text to truncate.
            max_tokens (int): Maximum number of tokens to keep.

        Returns:
            str: Truncated text.
        """
        tokens = self.tokenizer.encode(text = text)[:max_tokens]  # Tokenize and truncate
        truncated_text = self.tokenizer.decode(tokens)  # Decode back to string
        return truncated_text