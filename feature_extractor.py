# Import the specialized classes for each model type.
from models.align_feature_extractor import ALIGNFeatureExtractor
from models.blip2_feature_extractor import BLIP2FeatureExtractor
from models.clip_feature_extractor import CLIPFeatureExtractor
from models.longclip_feature_extractor import LongCLIPFeatureExtractor
from models.openclip_feature_extractor import OpenCLIPFeatureExtractor


def FeatureExtractor(model_name: str, checkpoint: str = None):
    """
    Function to create the correct feature extractor instance.

    Args:
        model_name (str): The name of the model to use.
        pretrained (str, optional): Pretrained weight specifier for models like LongCLIP.

    Returns:
        An instance of the appropriate FeatureExtractor subclass.

    Raises:
        ValueError: If an unsupported model name is provided or if LongCLIP is used without a checkpoint.
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
        return CLIPFeatureExtractor("openai/clip-vit-large-patch14")
    
    elif "longclip" in model_name:
        if checkpoint is None:
            raise ValueError("LongCLIP requires a checkpoint path.")
        return LongCLIPFeatureExtractor(checkpoint)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
