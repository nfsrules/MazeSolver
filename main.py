#!/usr/bin/env python
# coding: utf-8
# Author: Nelson Fernandez, Renault Research
# nelson.fernandez-pinto@renault.com

# Imports
from dataset import *
from architectures import *
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
                  alpha=1)

# Create CNN model
net = ConvNet()
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
batch_size = 1
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
epochs = 10
print('Losses and optimizer created...')


net.train()
print('Init training loop...')

for epoch in range(1, epochs+1):
    iters = 0
    for batch_idx, (data, target, path, goals, expert_flag) in enumerate(trainloader):
        # Convert X, Y to Torch variables
        grid = data   # save numpy version of maze grid

        # Convert input and target to tensors
        data, target = Variable(data), Variable(target)
        data, target = data.to(device).float(), target.to(device).float()

        # Set gradient to zero
        optimizer.zero_grad()

        # Get network output
        output = net(data)

        # Compute losses 
        if expert_flag:  # expert trajectory (imitation)
            loss = rmse(output, target)

        else:  # non-expert trajectory (environment).detach().numpy()
            # Maze walls loss (target, prediction)
            grid = grid[0][0]
            path = path[0]
            goals = goals[0]
            output = output.detach().numpy()

            # Reconstruct predicted grids
            pred_grid = torch.tensor(draw_solution_canvas(grid.detach().numpy(), output)).double()
            pred_path = torch.tensor(draw_solution_canvas(path.detach().numpy(), output)).double()
            pred_goals = torch.tensor(draw_goals_canvas(grid.detach().numpy(), output)).double()

            # Compute graphic losses
            # Wall loss
            wall_loss = graphic_loss(grid, pred_grid)
            # Path planning loss 
            path_loss = graphic_loss(path, pred_path)
            path_loss = abs(wall_loss-path_loss)
            # Goals loss
            goal_loss = graphic_loss(goals, pred_goals)

            # Final non-expert trajectory loss
            loss = Variable((wall_loss + path_loss + goal_loss), requires_grad=True)

        # Backpropagate loss & udpate gradient
        loss.backward()
        optimizer.step()

        # Print information
        iters += len(data)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, iters, len(trainloader.dataset),
                100. * (batch_idx + 1) / len(trainloader), loss.data.item()), end='\r', flush=True)
    print("")
                                 
                                 


