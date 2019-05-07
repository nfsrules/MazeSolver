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


def xy_transform(instance):
    '''Build X,Y pairs from Dataset class.
    X: 3 channel tensor
    Y: Center of the box
    '''
    X, path, goals, Y, expert_flag = instance

    # Correct Y to fixed lenght (50)
    ## This is a hack :((
    if Y.shape[0] <= 50:
        end = Y[-1]
        while Y.shape[0] <= 49:
            Y = np.vstack((Y, end))
    else:
        print('alert >50')

    return X.reshape(1, X.shape[0], X.shape[1]), Y, path, goals, expert_flag

# Generate a random maze each time you access the get_item method
# generate thousands of mazes during the init
# how to control if the maze is repeated?
# canvas_size: size of the desired maze
# nbr_instances: number of mazes to generate, default 3000
# difficulty: 'mixed', 'hard', 'medium', 'easy'
# nbr_trajectories

class MazeExplorer(Dataset):
    
    def __init__(self, maze_size=(64,64), nbr_instances=3000, difficulty='mixed', nbr_trajectories=20, generator='prims', solver='shortestpath', alpha=2):
        self.w, self.h = maze_size
        self.nbr_instances = nbr_instances
        self.difficulty = difficulty
        self.generator = generator
        self.solver = solver
        self.nbr_entrances = 3   # not sure, let's keep it fix
        self.nbr_trajectories = nbr_trajectories
        self.alpha = alpha

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

        # Number of valid/non valid trajectories to generate
        # We start from a 50%/50% assumption
        nbr_non_valid = int(self.nbr_trajectories/2)
        nbr_valid = int(self.nbr_trajectories - nbr_non_valid)

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
            # For each maze in 'aux' we generate nbr_trajectories versions
            # Each generated trajectory is a possible/ non possible solution
            # Case 1: Perturbated trajectories (controlled by alpha) / non-expert trajectories 
            #s_aux = [sibling_mazes(x, self.nbr_trajectories, self.w, self.h, self.alpha) for x in aux]
            s_aux_non_valid = [sibling_mazes(x, nbr_non_valid, self.w, self.h, self.alpha) for x in aux]
            s_aux_non_valid = np.concatenate(np.array(s_aux_non_valid), axis=0)

            # Case 2; Perturbating trajectories (slightly) still remains as expert trajectories
            # alpha=15 is normally enought for rejecting only 1% of the generated trajectories
            s_aux_valid = [sibling_mazes_expert(x, nbr_valid, self.w, self.h, 15) for x in aux]
            s_aux_valid = np.concatenate(np.array(s_aux_valid), axis=0)

            # Append family of mazes
            mazes.append(s_aux_non_valid)
            mazes.append(s_aux_valid)

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
        #solution = get_solution(maze['solutions'], maze['start'], maze['end'])        
        #solution = upsample_trajectory(solution, (self.w, self.h), maze['grid'].shape)

        # Estimate agent size in reference to road
        #road_width = estimate_road_width(path_planning)
          # Set agent size in reference to road width
        #agent_size = int(road_width/2)

        # Generate family of solutions / concatenate with solutions / only to visualize
        #family = generate_family_trajectories(solution, self.nbr_trajectories, road_width)
        
        return maze_grid, path_planning, goals, maze['upsampled_solution'], maze['expert']#, family


    def get_dopping_percentage(self):
        total_points = 0
        total_collisions = 0

        for i in tqdm(range(self.len())):
            grid, path, goals, solution, expert_flag = self[i]
            road_size = estimate_road_width(grid)
            #w, l = (int(road_size/2), int(road_size/2))
            collisions, total = check_trajectory(solution, grid, road_size)
            total_points = total_points + total
            total_collisions = total_collisions + collisions
            self.dopping = round((total_collisions/total_points)*100, 3)

        print('Estimated dopping percentage = ', self.dopping)
        return self.dopping


    def get_dopping_percentage_trajectories(self):
        dopping_counter = 0
        for i in tqdm(range(self.len())):
            grid, path, goals, solution, expert_flag = self[i]
            if expert_flag == False:
                dopping_counter = dopping_counter + 1
        self.dopping = round((dopping_counter/self.len())*100,3)
        print('Estimated dopping percentage = ', self.dopping)


    def len(self):
        return len(self.mazes)


    #def save(sel, directory='.'):
    #    self.mazes


