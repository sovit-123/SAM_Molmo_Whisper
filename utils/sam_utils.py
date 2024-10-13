"""
Helper functions for SAM visualization and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

# Helper functions for SAM2 segmentation map visualization.
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 40/255, 50/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        import cv2
        contours, _ = cv2.findContours(
            mask,cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(
                contour, epsilon=0.01, closed=True
            ) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, 
            contours, 
            -1, 
            (1, 0, 0, 1), 
            thickness=2
        ) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(
        pos_points[:, 0], 
        pos_points[:, 1], 
        color='green', 
        marker='.', 
        s=marker_size, 
        edgecolor='white', 
        linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], 
        neg_points[:, 1], 
        color='red', 
        marker='.', 
        s=marker_size, 
        edgecolor='white', 
        linewidth=1.25
    )   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle(
        (x0, y0), 
        w, 
        h, 
        edgecolor='green', 
        facecolor=(0, 0, 0, 0), 
        lw=2)
    )    

def show_masks(
    image, 
    masks, 
    scores, 
    point_coords=None, 
    box_coords=None, 
    input_labels=None, 
    borders=True
):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i == 0:  # Only show the highest scoring mask.
            show_mask(mask, plt.gca(), random_color=False, borders=borders)
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        show_box(box_coords, plt.gca())
    plt.axis('off')
    return plt