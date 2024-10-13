"""
General and miscellaneous helper utilities.
"""

import re
import numpy as np

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