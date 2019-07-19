#import sys,os
#sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.autograd import Variable  

import numpy as np
#from functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, prediction, target):
        return torch.sqrt(self.mse(prediction, target) + self.eps)

    
class GraphicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, prediction):
        add_grid = torch.add(target, prediction.t())
        return torch.sum(add_grid > 1.).float()

    
class GaussianLoss(nn.Module):
    '''Calculate collision Loss between ego and a bounding box.
    
    map_size: Image size Tuple example (150,150)
    height: Max value of the loss at the center of a bounding box
    step: Step resolution to calculate loss. Default 1 pixel
    narrow: Reduce variance of the Gaussian by 2 (95%). Default False
    plot: Plot the resulting loss function. Default False
    
    '''
    def __init__(self, map_size, height, step=1, narrow=False, plot=False):
        super().__init__()
        self.height = height 
        self.narrow = narrow
        self.plot = plot
        # Create a mesh gride of map size
        grid = torch.meshgrid([torch.arange(0, map_size[0], step),
                                 torch.arange(0, map_size[1], step)])
        # Torch meshgrid output is not the same as numpy  / correction
        self.grid = [grid[0][:,0].reshape(1,-1).float().to(device), 
                     grid[0][:,0].reshape(1,-1).t().float().to(device)]
    
    def forward(self, prediction, boxes):
        # Calculate a 2D Gaussian loss grid for each actor in the scene
        losses = []
        for box in boxes:
            # Unpack actor bounding box
            x0, y0 = torch.tensor(box['coordinates']).float().to(device)  # non normalized coordinates
            theta = torch.tensor(box['angle']).float().to(device) # In radians
            # We use the bounding box size as covariance in each axis
            sigma_X, h, sigma_Y = torch.tensor(box['whl']).float().to(device)  # box WHL in pixels
            
            if self.narrow:
                sigma_X = sigma_X/2
                sigma_Y = sigma_Y/2
            
            # Generate 2D-Gaussian loss custom parameters
            a = (torch.cos(theta)**2/(2*sigma_X**2)) + (torch.sin(theta)**2/(2*sigma_Y**2))
            b = (-1*torch.sin(2*theta)/(4*sigma_X**2)) + (torch.sin(2*theta)/(4*sigma_Y**2))
            c = (torch.sin(theta)**2/(2*sigma_X**2)) + (torch.cos(theta)**2/(2*sigma_Y**2))
            g = -1*(a*(self.grid[0]-x0)**2 + 2*b*(self.grid[0]-x0)*(self.grid[1]-y0) + c*(self.grid[1]-y0)**2)
            losses.append(self.height*torch.exp(g))
            
        # The final loss function is the superposition of individual losses    
        loss_function = torch.stack(losses, dim=0).sum(dim=0)
        
        if self.plot:
            plt.imshow(loss_function)
            for point in prediction:
                plt.scatter(point[0], point[1], marker='x', c='r')

        # For each prediction point estimate the loss due to each obstacle
        loss = [loss_function[point[1], point[0]] for point in prediction]
        print(loss)
        return torch.tensor(loss).mean()

    
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
                pred_loss.append(Variable(torch.tensor(0).float().to(device)))  # Do not calculate losses in this case

            else: # the  point is inside  the  canvas
                # Check if prediction is in drivable area
                if is_outside_road(hd_map, point): #  True if prediction is in non-drivable area
                    # Calculate loss (growing exponential)
                    # invert mask
                    image = hd_map*-1+1
                    # Find nearest drivable pixel
                    nonzero = (image.nonzero()).float().to(device)
                    # Calculate distances using L2
                    distances = torch.sqrt((nonzero[:,0]-point[0].float().to(device))** 2 + 
                                           (nonzero[:,1]-point[1].float().to(device))** 2)
                    
                    # Get shortest drivable point
                    nearest_index = torch.argmin(distances)
                    # find euclidean distance L2 (prediction and drivable point)
                    distance = torch.dist(nonzero[nearest_index], point.float().to(device))
                    # Calculate loss and append /  growing exponential
                    pred_loss.append(torch.exp(distance*(torch.log(torch.tensor(2).float().to(device))/self.k2)))

                else:  # prediction is inside drivable area
                    # Calculate loss (decreasing exponential)
                    # Find nearest non-drivable pixel
                    nonzero = (hd_map.nonzero()).float().to(device)
                    # Calculate distances L2
                    distances = torch.sqrt((nonzero[:,0]-point[0].float().to(device))** 2 + 
                                           (nonzero[:,1]-point[1].float().to(device))** 2)
                    
                    # Get shortest distance
                    nearest_index = torch.argmin(distances)
                    # find euclidean distance L2 (prediction and non-drivable point)
                    distance = torch.dist(nonzero[nearest_index], point.float().to(device))
                    # Calculate loss and append /  decreasing exponential
                    pred_loss.append(torch.exp(-1*distance**2/self.k1))

        return Variable(torch.tensor(pred_loss).to(device).mean())

    
def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


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
    neighbors = im[point[1]-1:point[1]+1, point[0]-1:point[0]+1].flatten()
    if 1 in neighbors:
        flag = True
    else:
        flag = False
    return flag
    

