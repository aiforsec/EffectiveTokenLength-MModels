import torch
from transformers import AutoProcessor, AutoModel

class FeatureExtractor:
    """
    Feature extractor for the model, supporting both image and text feature extraction.
    """
    def __init__(self, model_name: str):
        """
        Initializes the specified model and processor.
        
        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    @torch.no_grad()
    def extract_text_features(self, text: str):
        """
        Extracts and normalizes text features.
        
        Args:
            text (str): Input text to process.
        
        Returns:
            list: Normalized text feature vector.
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)
        return (text_features / text_features.norm(dim=-1, keepdim=True)).cpu().tolist()
    
    @torch.no_grad()
    def extract_image_features(self, image):
        """
        Extracts and normalizes image features.
        
        Args:
            image: Input image to process.
        
        Returns:
            list: Normalized image feature vector.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return (image_features / image_features.norm(dim=-1, keepdim=True)).cpu().tolist()
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncates input text to a specified number of tokens.
        
        Args:
            text (str): The text to truncate.
            max_tokens (int): Maximum number of tokens to retain.
        
        Returns:
            str: The truncated text.
        """
        tokenizer = self.processor.tokenizer
        tokens = tokenizer(text, truncation=False)["input_ids"][:max_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
