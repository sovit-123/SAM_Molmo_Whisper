import numpy as np
import torch
import re
import gradio as gr

from PIL import Image

from transformers import GenerationConfig
from utils.sam_utils import show_masks
from utils.load_models import (
    processor, molmo_model, transcriber, sam_predictor
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_coords(output_string, image):
    """
    Function to get x, y coordinates given Molmo model outputs.

    :param output_string: Output from the Molmo model.
    :param image: Image in PIL format.

    Returns:
        coordinates: Coordinates in format of [(x, y), (x, y)]
    """
    image = np.array(image)
    h, w = image.shape[:2]
    
    if 'points' in output_string:
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        coordinates = [(int(float(x_val)/100*w), int(float(y_val)/100*h)) for _, x_val, _, y_val in matches]
    else:
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [(int(float(match.group(1))/100*w), int(float(match.group(2))/100*h))]
    
    return coordinates


def get_output(image, prompt='Describe this image.'):
    """
    Function to get output from Molmo model given an image and a prompt.

    :param image: PIL image.
    :param prompt: User prompt.

    Returns:
        generated_text: Output generated by the model.
    """
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(molmo_model.device).unsqueeze(0) for k, v in inputs.items()}
    
    output = molmo_model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings='<|endoftext|>'),
        tokenizer=processor.tokenizer
    )
    
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

def process_image(image, prompt, audio):
    """
    Function combining all the components and returning the final 
    segmentation map.

    :param image: PIL image.
    :param prompt: User prompt.

    Returns:
        fig: Final segmentation map.
        prompt: Prompt from the Molmo model.
    """

    transcribed_text = ''

    if len(prompt) == 0:
        sr, y = audio
    
        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)
    
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        transcribed_text = transcriber({'sampling_rate': sr, 'raw': y})['text'] 
        prompt = transcribed_text

    print(prompt)

    # Get coordinates from the model output.
    output = get_output(image, prompt)
    coords = get_coords(output, image)
    
    # Prepare input for SAM
    input_points = np.array(coords)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    
    # Convert image to numpy array if it's not already.
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Predict mask.
    sam_predictor.set_image(image)
    with torch.no_grad():
        masks, scores, logits = sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
    
    # Sort masks by score.
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    
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
        title='Image Segmentation with SAM2, Molmo, and Whisper',
        description=f"Upload an image and provide a prompt to segment specific objects in the image. \
                    Text box input takes precedence. Text box needs to be empty to prompt via voice."
    )
    
    iface.launch(share=True)