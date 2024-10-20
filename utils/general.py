"""
General and miscellaneous helper utilities.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

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