"""
General and miscellaneous helper utilities.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from utils.sam_utils import show_video_mask

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
    elif 'point' in output_string:
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [(int(float(match.group(1))/100*w), int(float(match.group(2))/100*h))]
    else:
        return output_string
    
    return coordinates

def plot_image(image):
    """
    Converts a PIL image to Matplotlib plot.

    :param image: A PIL image.
    """
    image = np.array(image)

    dpi = plt.rcParams['figure.dpi']
    figsize = image.shape[1] / dpi, image.shape[0] / dpi

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    return plt

def extract_video_frame(video, path):
    """
    Function to extract video frames.

    :param video: The uploaded video.
    """
    print(video)
    cap = cv2.VideoCapture(video)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            file_name = f"{str(frame_index).zfill(5)}.jpg"

            cv2.imwrite(os.path.join(path, file_name), frame)

            frame_index += 1
        else:
            break

    cap.release()

def save_video(output_dir, w, h, video_frames_dir, frame_names, video_segments):
    """
    Function to save video results from SAM video predictor.
    """
    # OpenCV VideoWriter
    codec = cv2.VideoWriter_fourcc(*'VP90')
    # codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f"{output_dir}/molmo_points_output.webm",
        codec, 30,
        (w, h)
    )

    # Visualize a few segmentation result frames.
    vis_frame_stride = 1
    plt.close('all')

    dpi = plt.rcParams['figure.dpi']

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #### SAM visulization starts here ####
        image = Image.open(os.path.join(video_frames_dir, frame_names[out_frame_idx]))

        figsize = image.size[0] / dpi, image.size[1] / dpi
        plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(image)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_video_mask(out_mask, ax, obj_id=out_obj_id)

        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #### SAM visulization ends here ####
        
        #### Converting to Numpy and saving video starts here ####
        # Convert the Matplotlib plot to a NumPy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Get the RGBA buffer from the figure
        image_rgba = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)

        # Convert ARGB to RGBA
        image_rgba = np.roll(image_rgba, 3, axis=2)

        # Convert RGBA to RGB by discarding the alpha channel
        image_rgb = image_rgba[..., :3]

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Save the image using OpenCV
        out.write(image_bgr)

    # Close the plot to free memory
    plt.close(fig)
