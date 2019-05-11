#!/usr/bin/env python
# coding: utf-8
# Author: Nelson Fernandez, Renault Research
# nelson.fernandez-pinto@renault.com

import sys,os
sys.path.append(os.getcwd())

# Imports
from dataset import *
from architectures import *
#from functions import *
from losses import *
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)


# Generate dataset
me = MazeExplorer(maze_size=(64,64), nbr_instances=50, 
                  difficulty='easy', nbr_trajectories=20,
                  alpha=17)

# Show deopping percentages
me.get_dopping_percentage()  # point level
me.get_dopping_percentage_trajectories() # trajectory level

# Create CNN model
net = ConvNet().to(device)
print('CNN created...')


# Get set of train/test INDEXES (hacking SKlean train_test_split)
x = np.arange(0, me.len())
train_indexes, test_indexes, _, _ = train_test_split(x, x, 
                                                    test_size=0.33, 
                                                    random_state=7)


# Wrap transforming function to dataset object
t_me = TransformedDataset(me, xy_transform=xy_transform)

# Get training/test subsets
train_me = Subset(t_me, indices=train_indexes)
test_me = Subset(t_me, indices=test_indexes)

# Configure dataloaders
batch_size = 16
num_workers = 4

trainloader = DataLoader(train_me, shuffle=True, 
                          batch_size=batch_size, 
                          num_workers=num_workers, 
                          pin_memory=True)


testloader = DataLoader(test_me, shuffle=True, 
                         batch_size=batch_size, 
                         num_workers=1,
                         pin_memory=True)

print('Train test data loaders created...')

# Init losses
rmse = RMSELoss()
graphic_loss = GraphicLoss()

# Set optimizer (Adam)
optimizer = Adam(net.parameters(), lr=1e-3)
epochs = 200
print('Losses and optimizer created...')


net.train()
print('Init training loop...')

for epoch in range(1, epochs+1):
    iters = 0.0
    total_env = 0.0
    total_rmse = 0.0

    for batch_idx, (grid, target, path, goal, expert_flag) in enumerate(trainloader):
        # Convert X, Y to Torch variables
        #grid = data   # save numpy version of maze grid
        grid, target, path, goal, expert_flag = grid.to(device), target.to(device), path.to(device), goal.to(device), expert_flag.to(device)
        # Convert input and target to tensors
        grid, target = Variable(grid), Variable(target)
        grid, target = grid.to(device).float(), target.to(device).float()
 
        # Set gradient to zero
        optimizer.zero_grad()

        # Get network output
        output = net(grid)

        # RMSE loss / batch ready
        rmse_loss = rmse(output, target)

        # Grid Loss / Vectorized version
        # Generate grids from predictions 
        pred_grids = torch.tensor([draw_solution_canvas(x[0], y) for x in grid 
                                    for y in output]).to(device)

        pred_paths = torch.tensor([draw_solution_canvas(x, y) for x in path
                                    for y in output]).to(device)
        
        pred_goals = torch.tensor([draw_goals_canvas(x, y) for x in goal 
                                    for y in output]).to(device)

        # Compute losses
        wall_loss = [graphic_loss(y_true, y_pred) for y_true in grid for y_pred in pred_grids]
        path_loss = [graphic_loss(y_true, y_pred) for y_true in path for y_pred in pred_paths]
        goal_loss = [graphic_loss(y_true, y_pred) for y_true in goal for y_pred in pred_goals]

        env_loss = Variable((torch.tensor(wall_loss).mean() + 
                             torch.tensor(path_loss).mean() + 
                             torch.tensor(goal_loss).mean()), requires_grad=True)/10 #
 
        loss = rmse_loss + env_loss.to(device)
    
        # Backpropagate loss & udpate gradient
        loss.backward()
        optimizer.step()

        # Print information
        iters += loss
        total_env+= env_loss
        total_rmse+= rmse_loss

        print('Train Epoch: {} [{}/{} ({:.0f}%)] EnvLoss: {:.6f}  RMSELoss: {:.6f} TotalLoss: {:.6f}'.format(
                    epoch, batch_idx*batch_size, len(trainloader.dataset),
                    100. * (batch_idx) / len(trainloader), 
                    total_env/len(trainloader), 
                    total_rmse/len(trainloader),
                    iters/len(trainloader)
                    ), 
                    end='\r', flush=True)
    print("")                                 
                                     


