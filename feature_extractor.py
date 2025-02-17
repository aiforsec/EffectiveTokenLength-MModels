# feature_extractor.py


# Import the specialized classes

from models.align_feature_extractor import ALIGNFeatureExtractor
from models.blip2_feature_extractor import BLIP2FeatureExtractor
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.longclip_feature_extractor import LongCLIPFeatureExtractor
from models.openclip_feature_extractor import OpenCLIPFeatureExtractor


def FeatureExtractor(model_name: str, checkpoint: str = None):
    """
    Factory function to create the correct feature extractor instance.

    Args:
        model_name (str): The name of the model to use.
                          Examples:
                          - "kakaobrain/align-base"
                          - "Salesforce/blip2-opt-2.7b"
                          - "openai/clip-vit-large-patch14"
                          - "ViT-B-32" (for openclip)
        pretrained (str, optional): Pretrained weight specifier for models like OpenCLIP.

    Returns:
        An instance of the appropriate FeatureExtractor subclass.
    """
    model_name = model_name.lower()

    if "align" in model_name:
        return ALIGNFeatureExtractor()

    elif "blip2" in model_name:
        return BLIP2FeatureExtractor()

    elif "openclip" in model_name:
        return OpenCLIPFeatureExtractor()

    elif "clip_base" in model_name:
        return CLIPFeatureExtractor()
    
    elif "clip_large" in model_name:
        return CLIPFeatureExtractor("clip-vit-large-patch14")
    
    elif "longclip" in model_name:
        if checkpoint is None:
            raise ValueError("LongCLIP requires a checkpoint path.")
        return LongCLIPFeatureExtractor(checkpoint)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
