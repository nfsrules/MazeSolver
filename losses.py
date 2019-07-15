#import sys,os
#sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.autograd import Variable  

import numpy as np
#from functions import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(7)

#class RMSELoss(nn.Module):
#    def __init__(self, eps=1e-10):
#        super().__init__()
#        self.mse = nn.MSELoss()
#        self.eps = eps

#    def forward(self, prediction, target):
#        return torch.sqrt(self.mse(prediction, target) + self.eps)


class GraphicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, prediction):
        add_grid = torch.add(target, prediction.t())
        return torch.sum(add_grid > 1.).float()


#class OORLoss(nn.Module):
    '''Out of range Loss'''
#    def __init__(self, canvas_size=(64,64)):
#        super().__init__()
#        self.canvas_size = canvas_size[0]#

#    def forward(self, prediction, target):
#        for point in prediction:
#            x, y = point
#            if (x > self.canvas_size-1 or x < 0) or (y > self.canvas_size-1 or y <0):
#                loss = 0.5
#            else :
#                loss = 0.0
#        return loss


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


    
#def total_loss(rmse, graphic_loss, grid, path, goals, output, target, env=False):

    # Compute losses
#    loss = rmse(output, target)
    
#    if env:
#        rmse_loss = rmse(output, target)
        # non-expert trajectory (environment).detach().numpy()
        # Maze walls loss (target, prediction) / Convert to numpy arrays
#        grid = grid[0][0].detach().numpy()
#        path = path[0].detach().numpy()
#        goals = goals[0].detach().numpy()
#        output = output[0].detach().cpu().numpy()

        # Reconstruct predicted grids
#        pred_grid = torch.tensor(draw_solution_canvas(grid, output)).double().to(device)
#        pred_path = torch.tensor(draw_solution_canvas(path, output)).double().to(device)
#        pred_goals = torch.tensor(draw_goals_canvas(grid, output)).double().to(device)

        # Compute graphic losses
        # Wall loss
#        wall_loss = graphic_loss(torch.tensor(grid).to(device), pred_grid)
        # Path planning loss 
#        path_loss = graphic_loss(torch.tensor(path).to(device), pred_path)
#        path_loss = abs(wall_loss-path_loss)
        # Goals loss
#        goal_loss = graphic_loss(torch.tensor(goals).to(device), pred_goals)

        # Final non-expert trajectory loss
#        loss = Variable((wall_loss + path_loss + goal_loss), requires_grad=True)/10 + rmse_loss
        
#    return loss


#def total_loss_batch(rmse, graphic_loss, grid, path, goals, output, target):

    # Compute losses
#    print('output shape', output.shape)

#    rmse_loss = rmse(output.unsqueeze(0), target)

    # non-expert trajectory (environment).detach().numpy()
    # Maze walls loss (target, prediction) / Convert to numpy arrays
#    grid = grid[0][0].detach().numpy()
#    path = path[0].detach().numpy()
#    goals = goals[0].detach().numpy()
#    output = output.cpu().numpy()

    # Reconstruct predicted grids
#    pred_grid = torch.tensor(draw_solution_canvas(grid, output)).double()
#    pred_path = torch.tensor(draw_solution_canvas(path, output)).double()
#    pred_goals = torch.tensor(draw_goals_canvas(grid, output)).double()

    # Compute graphic losses
    # Wall loss
#    wall_loss = graphic_loss(grid, pred_grid)
    # Path planning loss 
#    path_loss = graphic_loss(path, pred_path)
#    path_loss = abs(wall_loss-path_loss)
    # Goals loss
#    goal_loss = graphic_loss(goals, pred_goals)

    # Final non-expert trajectory loss
#    loss = Variable((wall_loss + path_loss + goal_loss), requires_grad=True) + rmse_loss
    
#    return loss




#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#import numpy as np


def find_k(res=0.1 ,min_loss=0.01):
    '''Estimate K for a given attenuation 1% by default at 1 meter 
    from the border of the route.
      res: HD map resolution in pixel/meters
      min_loss: desired loss at d (0.01= 1% by default)
    '''
    return -1*(1/res)**2/np.log(min_loss)


def is_outside_frame(im, point):
    '''Check if prediction is outside the frame. This function
    DOES NOT take into account coordiantes normalization
    
    '''
    if (point[0] < 0) or (point[0] > im.shape[0]):
        flag = True
    elif (point[1] < 0) or (point[1] > im.shape[1]):
        flag = True
    else:
        flag = False
    return flag


def is_outside_road(im, point):
    '''Check if a point is outside drivable area.
    
    '''
    neighbors = im[point[0]-1:point[0]+1, point[1]-1:point[1]+1].flatten()
    if 1 in neighbors:
        flag = True
    else:
        flag = False
    return flag
    

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, prediction, target):
        return torch.sqrt(self.mse(prediction, target) + self.eps)
    

class RoadLoss(nn.Module):
    '''Calculate exponential custom losses.
    hd_map: 1 NON drivable / 0 Drivable  - binary mask
    prediction: list of prediction [x,2] / non normalized

    '''
    def __init__(self, k1=21.7, k2=40):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
    
    def forward(self, hd_map, prediction):
        # As one prediction is made of several points
        # we're  going to verify each point individually
        pred_loss = []
        for point in prediction:
            # Check if it is outside the frame
            if is_outside_frame(hd_map, point):  # True if the prediction is outside
                pred_loss.append(Variable(torch.tensor(0).float()))  # Do not calculate losses in this case

            else: # the  point is inside  the  canvas
                # Check if prediction is in drivable area
                if is_outside_road(hd_map, point): #  True if prediction is in non-drivable area
                    # Calculate loss (growing exponential)
                    # invert mask
                    image = hd_map*-1+1
                    # Find nearest drivable pixel
                    nonzero = (image.nonzero()).float()
                    # Calculate distances using L2
                    distances = torch.sqrt((nonzero[:,0]-point[0].float())** 2 + 
                                           (nonzero[:,1]-point[1].float())** 2)
                    
                    # Get shortest drivable point
                    nearest_index = torch.argmin(distances)
                    # find euclidean distance L2 (prediction and drivable point)
                    distance = torch.dist(nonzero[nearest_index], point.float())
                    # Calculate loss and append /  growing exponential
                    pred_loss.append(torch.exp(distance*(torch.log(torch.tensor(2).float())/self.k2))-1)

                else:  # prediction is inside drivable area
                    # Calculate loss (decreasing exponential)
                    # Find nearest non-drivable pixel
                    nonzero = (hd_map.nonzero()).float()
                    # Calculate distances L2
                    distances = torch.sqrt((nonzero[:,0]-point[0].float())** 2 + 
                                           (nonzero[:,1]-point[1].float())** 2)
                    
                    # Get shortest distance
                    nearest_index = torch.argmin(distances)
                    # find euclidean distance L2 (prediction and non-drivable point)
                    distance = torch.dist(nonzero[nearest_index], point.float())
                    # Calculate loss and append /  decreasing exponential
                    pred_loss.append(torch.exp(-1*distance**2/self.k1))

        return Variable(torch.tensor(pred_loss).mean())


