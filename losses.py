import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.autograd import Variable  
from functions import *
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


class OORLoss(nn.Module):
    '''Out of range Loss'''
    def __init__(self, canvas_size=(64,64)):
        super().__init__()
        self.canvas_size = canvas_size[0]

    def forward(self, prediction, target):
        for point in prediction:
            x, y = point
            if (x > self.canvas_size-1 or x < 0) or (y > self.canvas_size-1 or y <0):
                loss = 0.5
            else :
                loss = 0.0
        return loss


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


    
def total_loss(rmse, graphic_loss, grid, path, goals, output, target, env=False):

    # Compute losses
    loss = rmse(output, target)
    
    if env:
        rmse_loss = rmse(output, target)
        # non-expert trajectory (environment).detach().numpy()
        # Maze walls loss (target, prediction) / Convert to numpy arrays
        grid = grid[0][0].detach().numpy()
        path = path[0].detach().numpy()
        goals = goals[0].detach().numpy()
        output = output[0].detach().cpu().numpy()

        # Reconstruct predicted grids
        pred_grid = torch.tensor(draw_solution_canvas(grid, output)).double().to(device)
        pred_path = torch.tensor(draw_solution_canvas(path, output)).double().to(device)
        pred_goals = torch.tensor(draw_goals_canvas(grid, output)).double().to(device)

        # Compute graphic losses
        # Wall loss
        wall_loss = graphic_loss(torch.tensor(grid).to(device), pred_grid)
        # Path planning loss 
        path_loss = graphic_loss(torch.tensor(path).to(device), pred_path)
        path_loss = abs(wall_loss-path_loss)
        # Goals loss
        goal_loss = graphic_loss(torch.tensor(goals).to(device), pred_goals)

        # Final non-expert trajectory loss
        loss = Variable((wall_loss + path_loss + goal_loss), requires_grad=True)/10 + rmse_loss
        
    return loss


def total_loss_batch(rmse, graphic_loss, grid, path, goals, output, target):

    # Compute losses
    print('output shape', output.shape)

    rmse_loss = rmse(output.unsqueeze(0), target)

    # non-expert trajectory (environment).detach().numpy()
    # Maze walls loss (target, prediction) / Convert to numpy arrays
    grid = grid[0][0].detach().numpy()
    path = path[0].detach().numpy()
    goals = goals[0].detach().numpy()
    output = output.cpu().numpy()

    # Reconstruct predicted grids
    pred_grid = torch.tensor(draw_solution_canvas(grid, output)).double()
    pred_path = torch.tensor(draw_solution_canvas(path, output)).double()
    pred_goals = torch.tensor(draw_goals_canvas(grid, output)).double()

    # Compute graphic losses
    # Wall loss
    wall_loss = graphic_loss(grid, pred_grid)
    # Path planning loss 
    path_loss = graphic_loss(path, pred_path)
    path_loss = abs(wall_loss-path_loss)
    # Goals loss
    goal_loss = graphic_loss(goals, pred_goals)

    # Final non-expert trajectory loss
    loss = Variable((wall_loss + path_loss + goal_loss), requires_grad=True) + rmse_loss
    
    return loss