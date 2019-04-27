import numpy as np
import cv2


def overlap_loss(predicted_grid, grid):
    '''Calculate overlap loss between the agent and environement. It
    works for: mazes, path planning

    '''

    # Overlap predicted grid and environement grid
    add = cv2.add(grid.T, predicted_grid)
    # Check for colissions
    overlap = np.where(add == 2.)
    # Compute loss
    ## Number of overlaped pixels: m2 of overlapping ego + env
    if overlap[0].any(): # Overlap exist
        loss = len(overlap[0])  # Save number of colliding pixels as loss
    else:  # No overlap
        loss = 0
    ### Shall we add some kind of spatial smooth factor?
    ### Shall it be different on path planning & goals loss?
    return loss

