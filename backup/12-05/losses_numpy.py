import numpy as np
import cv2

# Nupy implementation of graphic Loss
def overlap_loss(predicted_grid, grid):
    '''Calculate overlap loss between the agent and environement. It
    works for: mazes, path planning
    NUMPY  implementation
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


def explorationLoss(target, prediction, expert_flag):
    ''' Compute Exploration loss (RMSE + Graphic Loss)
    Target: desired trajectory of points (maze solution)
    Prediction: predicted trajectory of points (predicted maze solution)
    Expert_flag: is the trajectory performed by an expert? 
    '''
    if expert_flag:  #  Trajectory performed by an expert, compute RMSE
        loss = rmse(target, prediction)
    else:  # trajectory is NOT expert, so do not imitate
        # Get graphic representation of predictions
        # Case 1: maze's grid (prediction points hit the walls?)
        predicted_grid = draw_solution_canvas(grid, prediction)
        target_grid = draw_solution_canvas(grid, target)
        # Case 2: maze's shorthest path (prediction points are inside the shortest path?)
        predicted_path = draw_solution_canvas(path, prediction)
        target_path = draw_solution_canvas(path, target)
        # Case 3: maze's goals (did the start and ending point overlaps with entrace and exit?)
        predicted_goals = draw_goals_canvas(grid, prediction)
        target_goals = draw_goals_canvas(grid, target)
        # The final loss is a linear combination of the environement losses
        # We multiply path and grid loss because they are highly correlated
        # Goals loss are usually way smaller than others, we multiply by 2
        loss = 0.5*(graphicLoss(target_grid, predicted_grid)) + 0.5*(graphicLoss(target_path, predicted_path)) + 2*(graphicLoss(target_goals, predicted_goals))
            
    return loss