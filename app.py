import numpy as np
import gradio as gr

from PIL import Image

from utils.sam_utils import show_masks
from utils.general import get_coords
from utils.model_utils import (
    get_whisper_output, get_molmo_output, get_sam_output
)

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
        transcribed_text, prompt = get_whisper_output(audio)

    print(prompt)

    # Get coordinates from the model output.
    output = get_molmo_output(image, prompt)
    coords = get_coords(output, image)
    
    # Prepare input for SAM
    input_points = np.array(coords)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    
    # Convert image to numpy array if it's not already.
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get SAM output
    masks, scores, logits, sorted_ind = get_sam_output(
        image, input_points, input_labels
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
        title='Image Segmentation with SAM2, Molmo, and Whisper',
        description=f"Upload an image and provide a prompt to segment specific objects in the image. \
                    Text box input takes precedence. Text box needs to be empty to prompt via voice."
    )
    
    iface.launch(share=True)