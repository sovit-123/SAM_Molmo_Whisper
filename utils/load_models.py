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

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

quant_config = BitsAndBytesConfig(load_in_4bit=True)

# Load SAM2 model.
sam_predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-large')

# Load Molmo model.
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924', 
    trust_remote_code=True, 
    device_map='auto', 
    torch_dtype='auto'
)
molmo_model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924', 
    trust_remote_code=True, 
    offload_folder='offload', 
    quantization_config=quant_config, 
    torch_dtype='auto'
)

# Load the Whisper model.
transcriber = pipeline(
    'automatic-speech-recognition',
    model='openai/whisper-small',
    device='cpu'
)