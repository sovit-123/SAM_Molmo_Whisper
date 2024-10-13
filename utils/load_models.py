"""
Code for loading SAM, Molmo, and Whisper models.
"""
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)

quant_config = BitsAndBytesConfig(load_in_4bit=True)

def load_sam(model_name='facebook/sam2.1-hiera-large', device='cpu'):
    """
    Load SAM2 model.
    """
    sam_predictor = SAM2ImagePredictor.from_pretrained(model_name)
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