"""
Code for loading SAM, Molmo, and Whisper models.
"""
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline,
    CLIPProcessor,
    CLIPModel,
    AutoModel
)

import spacy

quant_config = BitsAndBytesConfig(load_in_4bit=True)

def load_sam(model_name='facebook/sam2.1-hiera-large', device='cpu'):
    """
    Load SAM2 model.
    """
    sam_predictor = SAM2ImagePredictor.from_pretrained(model_name)
    return sam_predictor

def load_sam_video(model_name='facebook/sam2.1-hiera-tiny', device='cpu'):
    sam_predictor = SAM2VideoPredictor.from_pretrained(
        model_name
    )
    return sam_predictor

def load_molmo(model_name='allenai/MolmoE-1B-0924', device='cpu'):
    """
    Load Molmo model and processor.
    """
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map=device,
        torch_dtype='auto'
    )
    molmo_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        offload_folder='offload', 
        quantization_config=quant_config, 
        torch_dtype='auto',
        device_map=device
    )
    return processor, molmo_model

def load_whisper(model_name='openai/whisper-small', device='cpu'):
    """
    Load Whisper model.
    """
    transcriber = pipeline(
        'automatic-speech-recognition',
        model=model_name,
        device=device
    )
    return transcriber

def load_clip():
    """
    Loads the CLIP model for auto classification.
    """
    clip_model = CLIPModel.from_pretrained(
        'openai/clip-vit-base-patch32', device_map='cpu'
    )
    clip_processor = CLIPProcessor.from_pretrained(
        'openai/clip-vit-base-patch32'
    )
    return clip_processor, clip_model

def load_siglip():
    """
    Loads the CLIP model for auto classification.
    """
    siglip_model = AutoModel.from_pretrained(
        'google/siglip-so400m-patch14-384', device_map='cpu'
    )
    siglip_processor = AutoProcessor.from_pretrained(
        'google/siglip-so400m-patch14-384'
    )
    return siglip_processor, siglip_model

def load_spacy():
    """
    Loads the Spacy `en_core_web_sm` model for extracting nouns from 
    Molmo alt strings.
    """
    spacy_nlp = spacy.load('en_core_web_sm')
    return spacy_nlp