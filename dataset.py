import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import cv2
from mazelib import *
from functions import *
from tqdm import tqdm
import numpy as np
from tqdm import tqdm


class TransformedDataset(Dataset):
    
    def __init__(self, ds, xy_transform):
        self.ds = ds
        self.xy_transform = xy_transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.xy_transform(self.ds[index])


def xy_transform(dp):
    '''Build X,Y pairs from Dataset class.
    X: 3 channel tensor
    Y: Center of the box
    '''
    c, x, y, s = dp
    return x, y[0,:]

# Generate a random maze each time you access the get_item method
# generate thousands of mazes during the init
# how to control if the maze is repeated?
# canvas_size: size of the desired maze
# nbr_instances: number of mazes to generate, default 3000
# difficulty: 'mixed', 'hard', 'medium', 'easy'
# nbr_trajectories

class MazeExplorer(Dataset):
    
    def __init__(self, maze_size=(64,64), nbr_instances=3000, agent_size=(3,3), difficulty='mixed', nbr_trajectories=20, generator='prims', solver='shortestpath'):
        self.w, self.h = maze_size
        self.nbr_instances = nbr_instances
        self.difficulty = difficulty
        self.generator = generator
        self.solver = solver
        self.nbr_entrances = 3   # not sure, let's keep it fix
        self.agent_size = agent_size
        self.nbr_trajectories = nbr_trajectories

        if self.difficulty == 'mixed' or self.difficulty == 'hard' or self.difficulty == 'medium' or self.difficulty == 'easy':
            print('Creating MazeExplorer dataset of difficulty {}'.format(self.difficulty))
        else:
            raise ValueError('Difficulty not recognized.')

        if self.difficulty == 'mixed':
            self.difficulty = [(3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]
        elif self.difficulty == 'hard':
            self.difficulty = [(7,7), (8,8)]
        elif self.difficulty == 'medium':
            self.difficulty = [(5,5), (6,6)]
        elif self.difficulty == 'easy':
            self.difficulty = [(3,3), (4,4)]

        if self.generator == 'prims':
            print('Using Prims generator')
        else:
            raise ValueError('Generator not implemented.')

        if self.solver == 'shortestpath':
            print('Using shortest path solver')
        else:
            raise ValueError('Solver not implemented.')

        # Number of mazes by each difficulty 
        nbr_instances_by_difficulty = int(self.nbr_instances / len(self.difficulty))

        # Create one instance for each difficulty level
        mazes = []
        for size in self.difficulty:
            w, h = size
            m = Maze()
            m.generator = Prims(w, h)
            m.solver = ShortestPath()  #RandomMouse()#WallFollower() to implement later
            # Generate a group of unique mazes using MonteCarlo simulations
            aux = m.generate_monte_carlo(nbr_instances_by_difficulty, self.nbr_entrances)
            # Generate set of 'sibling' mazes with perturbed trajectories
            # One maze is converted to nbr_trajectories
            s_aux = [sibling_mazes(x, self.nbr_trajectories, self.w, self.h) for x in aux]
            s_aux = np.concatenate(np.array(s_aux), axis=0)
            # Append family of mazes
            mazes.append(s_aux)

        mazes = np.concatenate(np.array(mazes), axis=0)

        # Save mazes to RAM / possible as they are small
        self.mazes = mazes
        print('Dataset generated... {} available instances'.format(len(self.mazes)))


    def __getitem__(self, index):
        # Get  & pre-process current maze
        maze = self.mazes[index]
        maze_grid = clear_maze_entrances(maze['grid'], maze['start'], maze['end'])
        maze_grid = resize_grid(maze_grid['grid'], (self.w, self.h))
        
        # Generate path planning
        path_planning = generate_path_planning(maze['grid'], maze['start'], maze['end'], maze['solutions'])
        path_planning = resize_grid(path_planning['grid'], (self.w, self.h))

        # Generate goals (entrance/exit) layer
        goals = generate_entrances(maze['grid'], maze['start'], maze['end'])
        goals = resize_grid(goals['grid'], (self.w, self.h))

        # Generate maze's solution / The solution is a sequence with all points (start to end)
        solution = get_solution(maze['solutions'], maze['start'], maze['end'])        
        solution = upsample_trajectory(solution, (self.w, self.h), maze['grid'].shape)

        # Estimate agent size in reference to road
        road_width = estimate_road_width(path_planning)

        # Generate family of solutions / concatenate with solutions / only to visualize
        family = generate_family_trajectories(solution, self.nbr_trajectories, road_width)
        
        # Set agent size in reference to road width
        agent_size = int(road_width/2)
        #agent_size = (agent_size, agent_size)

        return maze_grid, path_planning, goals, agent_size, maze['upsampled_solution'], family


    def get_dopping_percentage(self):
        total_points = 0
        total_collisions = 0

        for i in tqdm(range(self.len())):
            grid, path, goals, agent_size, solution, family = self[i]
            collisions, total = check_trajectory(solution, grid, agent_size)
            total_points = total_points + total
            total_collisions = total_collisions + collisions
            self.dopping = round((total_collisions/total_points)*100, 3)
            
        print('Estimated dopping percentage = ', self.dopping)

        return self.dopping


    def len(self):
        return len(self.mazes)
