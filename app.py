import numpy as np
import gradio as gr
import torch
import os
import shutil
import gc

from PIL import Image, ImageDraw
from utils.sam_utils import show_masks
from utils.general import (
    get_coords, 
    plot_image, 
    extract_video_frame,
    save_video,
)
from utils.model_utils import (
    get_whisper_output, get_molmo_output, get_sam_output, get_spacy_output
)
from utils.load_models import (
    load_molmo, load_sam, load_whisper, load_sam_video, load_siglip, load_spacy
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

molmo_model_name = None
sam_model_name = None
whisper_model_name = None
processor, molmo_model = None, None
sam_predictor = None
transcriber = None

def process_image(
    image_path, 
    prompt, 
    audio,
    whisper_tag,
    molmo_tag,
    sam_tag,
    clip_label,
    draw_bbox,
    sequential_processing,
    random_color,
    chat_only
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
    :param clip_label: Whether to enable auto labeling using CLIP.

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
    global clicked_points

    coords = []

    image = Image.open(image_path)

    # Check if user chose different model, and load appropriately.
    if molmo_tag != molmo_model_name:
        gr.Info(message=f"Loading {molmo_tag}", duration=20)
        processor, molmo_model = load_molmo(model_name=molmo_tag, device=device)
        molmo_model_name = molmo_tag

    if not chat_only: # Load SAM only if `chat_only` mode is not selected.
        if sam_tag != sam_model_name:
            gr.Info(message=f"Loading {sam_tag}", duration=20)
            sam_predictor = load_sam(model_name=sam_tag)
            sam_model_name = sam_tag

    if audio is not None: # Load Whisper only if audio prompt is present.
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

    molmo_output = get_coords(output, image)

    if type(molmo_output) == str and len(clicked_points) == 0: # If we get image caption instead of points.
        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return  plot_image(image), output, transcribed_text
    
     # There is a chance the user clicks points and Molmo outputs string. In
     # that case, we do not want to append the Molmo string output to `coords`
     # but only the clicked points.
    if type(molmo_output) != str:
        coords.extend(molmo_output)
    
    # Append the clicked points to `coords`.
    coords.extend(clicked_points)
    
    # Load CLIP and Spacy models if `clip_label` is True.
    if clip_label:
        spacy_nlp = load_spacy()
        # clip_processor, clip_model = load_clip()
        clip_processor, clip_model = load_siglip()

        # Get the nouns list.
        nouns = get_spacy_output(outputs=output, model=spacy_nlp)
    
    # Prepare input for SAM
    input_points = np.array(coords)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    
    # Convert image to numpy array if it's not already.
    if isinstance(image, Image.Image):
        image = np.array(image)

    if chat_only:
        gr.Warning('Chat Only mode chosen. Ignoring every other option.')
    
    if not chat_only and clip_label: # If CLIP auto-labelling is enabled.
        label_array = [] # To store CLIP label after each loop.
        final_mask = np.zeros_like(image.transpose(2, 0, 1), dtype=np.float32)

        # This probably takes as many times longer as the number of objects
        # detected by Molmo.
        for input_point, input_label in zip(input_points, input_labels):
            masks, scores, logits, sorted_ind = get_sam_output(
                image,
                sam_predictor,
                input_points=[input_point],
                input_labels=[input_label]
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            final_mask += masks
        
            masks_copy = masks.copy()
            masks_copy = masks_copy.transpose(1, 2, 0)

            masked_image = (image * np.expand_dims(masks_copy[:, :, 0], axis=-1))
            masked_image = masked_image.astype(np.uint8)

            # Process masked image and give input to CLIP.
            clip_inputs = clip_processor(
                text=nouns, 
                images=Image.fromarray(masked_image), 
                return_tensors='pt', 
                padding=True
            )
            clip_outputs = clip_model(**clip_inputs)
            clip_logits_per_image = clip_outputs.logits_per_image # this is the image-text similarity score
            clip_probs = clip_logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
            clip_label = nouns[np.argmax(clip_probs.detach().cpu())]

            label_array.append(clip_label)

        im = final_mask >= 1
        final_mask[im] = 1
        final_mask[np.logical_not(im)] = 0
        
        fig = show_masks(
            image, 
            final_mask, 
            scores, 
            point_coords=input_points, 
            input_labels=input_labels, 
            borders=True,
            clip_label=label_array,
            draw_bbox=draw_bbox,
            random_color=random_color
        )

        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return fig, output, transcribed_text
    
    if not chat_only and  sequential_processing: # If sequential processing of points is enabled without CLIP.
        final_mask = np.zeros_like(image.transpose(2, 0, 1), dtype=np.float32)

        # This probably takes as many times longer as the number of objects
        # detected by Molmo.
        for input_point, input_label in zip(input_points, input_labels):
            masks, scores, logits, sorted_ind = get_sam_output(
                image,
                sam_predictor,
                input_points=[input_point],
                input_labels=[input_label]
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            final_mask += masks
        
            masks_copy = masks.copy()
            masks_copy = masks_copy.transpose(1, 2, 0)

            masked_image = (image * np.expand_dims(masks_copy[:, :, 0], axis=-1))
            masked_image = masked_image.astype(np.uint8)

        im = final_mask >= 1
        final_mask[im] = 1
        final_mask[np.logical_not(im)] = 0
        
        fig = show_masks(
            image, 
            final_mask, 
            scores, 
            point_coords=input_points, 
            input_labels=input_labels, 
            borders=True,
            draw_bbox=draw_bbox,
            random_color=random_color
        )
        
        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return fig, output, transcribed_text
    
    else:
        masks, scores, logits, sorted_ind = None, None, None, None
        if not chat_only: # Get SAM output
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
            borders=True,
            draw_bbox=draw_bbox,
            chat_only=chat_only
        )
        
        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return fig, output, transcribed_text

def process_video(
    video, 
    prompt, 
    audio,
    whisper_tag,
    molmo_tag,
    sam_tag
):
    """
    Function combining all the components and returning the final 
    segmentation map.

    :param video: A .avi or .mp4 video.
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

    coords = []

    sam_device_string = 'cuda'

    extract_video_frame(video=video, path=temp_dir)

    # Check if user chose different model, and load appropriately.
    # Fow now, load the Molmo model every time as we are deleting it later on.
    # if molmo_tag != molmo_model_name:
    gr.Info(message=f"Loading {molmo_tag}", duration=20)
    processor, molmo_model = load_molmo(model_name=molmo_tag, device=device)
    molmo_model_name = molmo_tag

    if whisper_tag != whisper_model_name:
        gr.Info(message=f"Loading {whisper_tag}", duration=20)
        transcriber = load_whisper(model_name=whisper_tag, device='cpu')
        whisper_model_name = whisper_tag

    transcribed_text = ''

    if len(prompt) == 0:
        transcribed_text, prompt = get_whisper_output(audio, transcriber)

    # Get the first frame from the extracted videos.
    image = Image.open('temp/00000.jpg')

    # Get coordinates from the model output.
    output = get_molmo_output(
        image, 
        processor,
        molmo_model,
        prompt
    )

    # Delete the Molmo model to free GPU memory.
    del processor, molmo_model
    gc.collect()
    torch.cuda.empty_cache()

    if sam_tag != sam_model_name:
        gr.Info(message=f"Loading {sam_tag}", duration=20)
        sam_predictor = load_sam_video(model_name=sam_tag, device=sam_device_string)
        sam_model_name = sam_tag

    # Reset SAM state and initialize the predictor with frames.
    try:
        sam_predictor.reset_state(inference_state)
    except:
        with torch.inference_mode(), torch.autocast(device_type=sam_device_string, dtype=torch.bfloat16):
            inference_state = sam_predictor.init_state(video_path=temp_dir)

    print(prompt)

    molmo_output = get_coords(output, image)
    coords.extend(molmo_output)

    if type(coords) == str: # If we get image caption instead of points.
        return  plot_image(image), output, transcribed_text
    
    # Prepare input for SAM
    input_points = np.array(coords)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    
    # Adding the points to the first frame.
    for i in range(len(input_points)):
        input_point = np.array([input_points[i]])
        input_label = np.array([input_labels[i]])
        ann_frame_idx = 0 # Frame index to interact/start with.
        ann_object_id = i # Give a unique object ID to the object, an integer.

        with torch.inference_mode(), torch.autocast(device_type=sam_device_string, dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = sam_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_object_id,
                points=input_point,
                labels=input_label
            )

    # Propagate through the entire video.
    # Propgate the prompt to get masklet across the video.
    # Run propagation throughout the video and collect the results in a dict
    video_segments = {}  # `video_segments` contains the per-frame segmentation results
    max_frame_num_to_track = None
    with torch.inference_mode(), torch.autocast(device_type=sam_device_string, dtype=torch.bfloat16):
        for out_frame_idx, out_obj_ids, out_mask_logits in sam_predictor.propagate_in_video(
            inference_state, max_frame_num_to_track=max_frame_num_to_track
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    # Get the frame names for saving the final video.
    frame_names = [
        p for p in os.listdir(temp_dir)
        if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Save video.
    w, h = image.size
    save_video(output_dir, w, h, temp_dir, frame_names, video_segments)
    
    return os.path.join(output_dir, 'molmo_points_output.webm'), output, transcribed_text

# Global list to store clicked points
clicked_points = []

def draw_circle_on_img(img, center, radius=10, color=[255, 0, 0]):
    """Draw a circle on the image with the given radius and color."""
    x, y = center
    point_radius, point_color = 5, (255, 255, 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [
            (x - point_radius, y - point_radius), 
            (x + point_radius, y + point_radius)
        ],
        fill=point_color
    )
    return img

def get_click_coords(img, evt: gr.SelectData):
    """Handle the click event and draw a circle at the clicked coordinates."""
    global clicked_points

    out = Image.open(img)
    clicked_points.append((evt.index[0], evt.index[1]))
    
    for point in clicked_points:
        out = draw_circle_on_img(out, point)

    return out


with gr.Blocks(
    title='Image Segmentation with SAM2, Molmo, and Whisper'
) as image_interface:
    # Inputs.
    img_input = gr.Image(type='filepath', label='Upload Image')
    txt_input = gr.Textbox(label='Prompt', placeholder='e.g., Point where the dog is.')
    audio_input = gr.Audio(sources=['microphone'])
    
    # Outputs.
    img_plt_out = gr.Plot(label='Segmentation Result', format='png')
    molmo_out = gr.Textbox(label='Molmo Output')
    whisper_out = gr.Textbox(label='Whisper Output')

    with gr.Row():
        with gr.Column():
            pointed_image = gr.Image(type='pil', label='Image with points')

    img_input.select(get_click_coords, [img_input], [pointed_image])
    
    # Additional inputs.
    whisper_models = gr.Dropdown(
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
    )

    molmo_models = gr.Dropdown(
        label='Molmo Models',
        choices=(
            'allenai/MolmoE-1B-0924',
            'allenai/Molmo-7B-O-0924',
            'allenai/Molmo-7B-D-0924',
            'allenai/Molmo-72B-0924',
        ),
        value='allenai/MolmoE-1B-0924'
    )

    sam_models = gr.Dropdown(
        label='SAM Models',
        choices=(
            'facebook/sam2.1-hiera-tiny',
            'facebook/sam2.1-hiera-small',
            'facebook/sam2.1-hiera-base-plus',
            'facebook/sam2.1-hiera-large',
        ),
        value='facebook/sam2.1-hiera-large'
    )

    clip_checkbox = gr.Checkbox(
        value=False, 
        label='Enable CLIP Auto Labelling.',
        info='Slower but gives better segmentations maps along with labels'
    )

    bbox_checkbox = gr.Checkbox(
        value=False, 
        label='Draw Bounding Boxes',
        info='Whether to draw bounding boxes around the segmentation objects. \
            Works best with CLIP Auto Labelling.'
    )

    seq_proc_checkbox = gr.Checkbox(
        value=False, 
        label='Sequential Processing',
        info='Process Molmo points sequentially generating one mask at a time. \
            Slower but more accurate masks.'
    )

    rnd_col_mask_checkbox = gr.Checkbox(
        value=False,
        label='Random color Mask',
        info='Randomly choose a mask color.'
    )

    chat_only_checkbox = gr.Checkbox(
        value=False,
        label='Chat Only Mode',
        info='If wanting pointing and chatting only without SAM \
            segmentation (saves inference time and memory). All other \
            above checkboxes will be ignored.'
    )

    gr.Interface(
        fn=process_image,
        inputs=[
            img_input, txt_input, audio_input
        ],
        outputs=[
            img_plt_out, molmo_out, whisper_out
        ],
        additional_inputs=[
            whisper_models,
            molmo_models,
            sam_models,
            clip_checkbox,
            bbox_checkbox,
            seq_proc_checkbox,
            rnd_col_mask_checkbox,
            chat_only_checkbox
        ],
        description=f"Upload an image and provide a prompt to segment specific objects in the image. \
                    Text box input takes precedence. Text box needs to be empty to prompt via voice."
    )

with gr.Blocks(
    title='Video Segmentation with SAM2, Molmo, and Whisper'
) as video_interface:
    # Inputs.
    vid_input = gr.Video(label='Upload Image')
    txt_input = gr.Textbox(label='Prompt', placeholder='e.g., Point where the dog is.')
    audio_input = gr.Audio(sources=['microphone'])

    # Outputs.
    vid_out = gr.Video(label='Segmentation Result', format='webm')
    molmo_out = gr.Textbox(label='Molmo Output')
    whisper_out = gr.Textbox(label='Whisper Output')
    
    # Additional inputs.
    whisper_models = gr.Dropdown(
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
    )

    molmo_models = gr.Dropdown(
        label='Molmo Models',
        choices=(
            'allenai/MolmoE-1B-0924',
            'allenai/Molmo-7B-O-0924',
            'allenai/Molmo-7B-D-0924',
            'allenai/Molmo-72B-0924',
        ),
        value='allenai/MolmoE-1B-0924'
    )

    sam_models = gr.Dropdown(
        label='SAM Models',
        choices=(
            'facebook/sam2.1-hiera-tiny',
            'facebook/sam2.1-hiera-small',
            'facebook/sam2.1-hiera-base-plus',
            'facebook/sam2.1-hiera-large',
        ),
        value='facebook/sam2.1-hiera-large'
    )

    gr.Interface(
        fn=process_video,
        inputs=[
            vid_input, txt_input, audio_input
        ],
        outputs=[
            vid_out, molmo_out, whisper_out
        ],
        additional_inputs=[
            whisper_models,
            molmo_models,
            sam_models
        ],
        description=f"Upload a video and provide a prompt to segment specific objects in the video. \
                    Text box input takes precedence. Text box needs to be empty to prompt via voice."
    )

if __name__ == '__main__':
    # A temporary directory to save extracted frames for video segmentation.
    temp_dir = 'temp'
    try:
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    except:
        os.makedirs(temp_dir, exist_ok=True)

    # An output directory to save results.
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Gradio interface.
    with gr.Blocks() as iface:
        with gr.Tab('Image processing'):
            image_interface.render()
        
        with gr.Tab('Video Processing'):
            video_interface.render()

    # iface = gr.TabbedInterface(
    #     [
    #         image_interface, 
    #         video_interface,
    #         demo
    #     ],
    #     tab_names=[
    #         'Image processing',
    #         'Video processing',
    #         'Point processor'
    #     ]
    # )
    
    iface.launch(share=True)