import numpy as np
import gradio as gr
import torch
import matplotlib.pyplot as plt

from PIL import Image
from utils.sam_utils import show_masks
from utils.general import get_coords, plot_image
from utils.model_utils import (
    get_whisper_output, get_molmo_output, get_sam_output
)
from utils.load_models import (
    load_molmo, load_sam, load_whisper
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

molmo_model_name = None
sam_model_name = None
whisper_model_name = None
processor, molmo_model = None, None
sam_predictor = None
transcriber = None

def process_image(
    image, 
    prompt, 
    audio,
    whisper_tag,
    molmo_tag,
    sam_tag
):
    """
    Function combining all the components and returning the final 
    segmentation map.

    :param image: PIL image.
    :param prompt: User prompt.
    :param audio: The audio command from user.
    :param molmo_tag: Molmo Hugging Face model tag.
    :param sam_tag: SAM Hugging Face model tag.
    :param Whisper_tag: Whisper Hugging Face model tag.

    Returns:
        fig: Final segmentation map.
        prompt: Prompt from the Molmo model.
        transcribed_test: The Whisper transcribed text.
    """

    global molmo_model_name
    global sam_model_name
    global whisper_model_name
    global processor
    global molmo_model
    global sam_predictor
    global transcriber

    # Check if user chose different model, and load appropriately.
    if molmo_tag != molmo_model_name:
        gr.Info(message=f"Loading {molmo_tag}", duration=20)
        processor, molmo_model = load_molmo(model_name=molmo_tag, device=device)
        molmo_model_name = molmo_tag

    if sam_tag != sam_model_name:
        gr.Info(message=f"Loading {sam_tag}", duration=20)
        sam_predictor = load_sam(model_name=sam_tag)
        sam_model_name = sam_tag

    if whisper_tag != whisper_model_name:
        gr.Info(message=f"Loading {whisper_tag}", duration=20)
        transcriber = load_whisper(model_name=whisper_tag, device='cpu')
        whisper_model_name = whisper_tag

    transcribed_text = ''

    if len(prompt) == 0:
        transcribed_text, prompt = get_whisper_output(audio, transcriber)

    print(prompt)

    # Get coordinates from the model output.
    output = get_molmo_output(
        image, 
        processor,
        molmo_model,
        prompt
    )

    coords = get_coords(output, image)

    if type(coords) == str: # If we get image caption instead of points.
        return  plot_image(image), output, transcribed_text
    
    # Prepare input for SAM
    input_points = np.array(coords)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    
    # Convert image to numpy array if it's not already.
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get SAM output
    masks, scores, logits, sorted_ind = get_sam_output(
        image, sam_predictor, input_points, input_labels
    )
    
    # Visualize results.
    fig = show_masks(
        image, 
        masks, 
        scores, 
        point_coords=input_points, 
        input_labels=input_labels, 
        borders=True
    )
    
    return fig, output, transcribed_text

if __name__ == '__main__':
    # Gradio interface.
    iface = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type='pil', label='Upload Image'),
            gr.Textbox(label='Prompt', placeholder='e.g., Point where the dog is.'),
            gr.Audio(sources=['microphone'])
        ],
        outputs=[
            gr.Plot(label='Segmentation Result', format='png'),
            gr.Textbox(label='Molmo Output'),
            gr.Textbox(label='Whisper Output'),
        ],
        additional_inputs=[
            gr.Dropdown(
                label='Whisper Models',
                choices=(
                    'openai/whisper-tiny',
                    'openai/whisper-base',
                    'openai/whisper-small',
                    'openai/whisper-medium',
                    'openai/whisper-large-v3',
                    'openai/whisper-large-v3-turbo',
                ),
                value='openai/whisper-small'
            ),
            gr.Dropdown(
                label='Molmo Models',
                choices=(
                    'allenai/MolmoE-1B-0924',
                ),
                value='allenai/MolmoE-1B-0924'
            ),
            gr.Dropdown(
                label='SAM Models',
                choices=(
                    'facebook/sam2.1-hiera-tiny',
                    'facebook/sam2.1-hiera-small',
                    'facebook/sam2.1-hiera-base-plus',
                    'facebook/sam2.1-hiera-large',
                ),
                value='facebook/sam2.1-hiera-large'
            ),
        ],
        title='Image Segmentation with SAM2, Molmo, and Whisper',
        description=f"Upload an image and provide a prompt to segment specific objects in the image. \
                    Text box input takes precedence. Text box needs to be empty to prompt via voice."
    )
    
    iface.launch(share=True)