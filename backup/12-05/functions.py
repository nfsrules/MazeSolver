from mazelib import *
import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
import cv2
import math
from skimage.transform import resize
from IPython import display
import time
from losses import *
from losses_numpy import *


# List of maze sizes
# The bigger the harder
small = [(3,3), (4,4)]    # upsampling factor 3
medium = [(5,5), (6,6)]   # upsampling factor 2
large = [(7,7), (8,8)]    # upsampling factor 1.5
    

def generate_mazes(maze_size, nbr_mazes, nbr_entrances):
    '''Generate family of unique mazes of the same size.
    maze: maze instance
    nbr_mazes: number of mazes to generate
    nbr_entrances: number of random entrances to generate
    returns
        maze grid, start, ending points and solution (list)
    '''
    # Unpack maze size
    w, h = maze_size
    # Create maze instace
    m = Maze()
    # Set a generator
    m.generator = Prims(w,h)
    # Set a solver
    m.solver = shortestPath()
    
    return m.generate_monte_carlo(nbr_mazes, nbr_entrances)


def pick_by_difficulty(mazes, difficulty):
    '''Select a given maze from a list based on its difficulty.
    (length of the solution)
    mazes: list of generated mazes
    difficulty: float [0, 1.] being 1 the longest path and 0 the shortest
    '''
    
    if difficulty < 0.0 or difficulty > 1.0:
        raise ValueError('Maze difficulty must be set from 0 to 1.') 
            
    # based on optional parameter, choose the maze of the correct difficulty
    posi = int((len(mazes) - 1) * difficulty)

    # save final results of Monte Carlo Simulations to this object
    grid = mazes[posi]['grid']
    start = mazes[posi]['start']
    end = mazes[posi]['end']
    solutions = mazes[posi]['solutions']
    
    return {'grid': grid, 'start':start, 'end':end, 'solution':solutions}
    
    
def sol2numpy(solution):
    '''Unpack maze solution to numpy array'''
    x = []
    y = []
    for element in solution[0]:
        x_, y_ = element
        x.append(x_)
        y.append(y_)
    return np.stack((x,y), axis=1)


def clear_maze_entrances(grid, start, end):
    '''Clear maze entrances setting first channel
    to 0 (0 means drivable)
    grid: input maze numpy array
    start, end: maze's entrance and exit list
    Returns grid
    '''
    # Clear entrances / Set entrances to 0 (drivable)
    grid[start[0], start[1]] = 0
    grid[end[0], end[1]] = 0        

    return {'grid':grid}


def get_solution(solution, start, end):
    '''Get solution sequence as numpy array

    '''
    start = np.array(start)
    end = np.array(end)
    solution = sol2numpy(solution)  # unpack solution
    solution = np.concatenate((start.reshape(1,2), solution, 
                                end.reshape(1,2)), axis=0)
    return solution


def generate_path_planning(grid, start, end, solution):
    '''Generate path planning canvas 
    
    '''
    start = np.array(start)
    end = np.array(end)
    # Create empty canvas (1: non drivable)
    canvas = np.ones_like(grid)
    # Unpack solution
    solution = sol2numpy(solution)
    # and start and end to solution
    solution = np.concatenate((start.reshape(1,2), solution, 
                                end.reshape(1,2)), axis=0)
    # Paint path planning solution on canvas (0 drivable)
    for pixel in solution:
        canvas[pixel[0], pixel[1]]= 0
        
    return {'grid':canvas, 'solution':solution}
    
    
def generate_entrances(grid, start, end):
    '''Generates a canvas with maze inputs and outputs
    
    '''
    # Create empty canvas (1: non drivable)
    canvas = np.ones_like(grid)

    # Paint start and end as drivable
    canvas[start[0], start[1]] = 0
    canvas[end[0], end[1]] = 0
        
    return {'grid':canvas}


def polygon_from_ego(center, angle=90, l=17.3, w=41.0):
    '''Draw box of ego in reference to UPSAMPLED maze.
    px, py: coordinated of the center of ego
    angle: heading angle of ego (degrees)
    width, leght: agent size

    '''
    # Unpacking egopose as center of the vehicle (original map)
    cx, cy = center
    # Convert heading angle to radians
    theta = math.radians(angle)

    # Generate corners of the bounding box
    bbox = np.matlib.repmat([[cx], [cy]], 1, 5) + \
       np.matmul([[math.cos(theta), math.sin(theta)],
                  [-math.sin(theta), math.cos(theta)]],
                 [[-w / 2, w / 2, w / 2, -w / 2, w / 2 + 8],
                  [-l / 2, -l / 2, l / 2, l / 2, 0]])
    
    # add first point
    x1, y1 = bbox[0][0], bbox[1][0]
    # add second point
    x2, y2 = bbox[0][1], bbox[1][1]
    # add forth point
    x3, y3 = bbox[0][2], bbox[1][2]
    # add fifth point
    x4, y4 = bbox[0][3], bbox[1][3]
    
    polygon = [x1, y1, x2, y2, x3, y3, x4, y4]

    return np.array(polygon).reshape((-1,1,2)).astype(np.int32)


def draw_egopose(grid, polygon, ann=True):
    '''Draw box with egopose on a canvas
    0: agent is not present
    1: agent is present
    '''
    canvas = np.zeros_like(grid)  # 0: agent not present
    return cv2.drawContours(canvas, [polygon], 0, (1,0,0), -1)  #  1: agent present (polygon)  


def resize_grid(grid, size=(64,64)):
    v = resize(grid, size, order=0)
    return (v - v.min()) / (v.max() - v.min())


def upsample_trajectory(solution, final_size, init_size):
    
    upsampling_factor = final_size[0] / init_size[0]
    
    if init_size[0] == 7:
        correction = 4
    elif init_size[0] == 9:
        correction = 3.5
    elif init_size[0] == 11:
        correction = 2.5
    elif init_size[0] == 13:
        correction = 2
    elif init_size[0] == 15 or init_size[0] == 17:
        correction = 1.5
    else:
        raise ValueError('Size not implemented...')
    
    return solution * upsampling_factor + correction


def estimate_road_width(path_planning):
    '''Estimate the width of the drivable area. This quantity 
    is used to limit the intensity of the trajectory
    perturbations
    
    '''
    w, h = path_planning.shape
    
    road_size_x = []
    road_size_y = []

    for i in range(w):
        sec = path_planning[i, :]  # get section
        rows = np.where(sec == 0)  # where is the road?  

        if rows[0].any():
            max_ = np.max(rows) - np.min(rows)  # max road size in pixels
            road_size_x.append(max_)
        else:
            road_size_x.append(0)

    for i in range(w):
        sec = path_planning[:, i]  # get section
        rows = np.where(sec == 0)  # where is the road?  

        if rows[0].any():
            max_ = np.max(rows) - np.min(rows)  # max road size in pixels
            road_size_y.append(max_)
        else:
            road_size_y.append(0)

    return np.unique(np.stack((road_size_x, road_size_y)))[1]


def perturbate(mean, road_size, nbr_samples=1, alpha=2):
    '''Perturbate point with 2D noise of mean (px,py)
    and covariance road_size/2 in both axis
    '''
    cov = [[road_size/alpha, 0],[0, road_size/alpha]]
    x, y = np.random.multivariate_normal(mean, cov, nbr_samples).T
    return np.array([x, y]).reshape(-1,2)


def perturbate_trajectory(solution, road_size, nbr_samples=1, alpha=2):
    '''Generate a perturbated trajectory from an example
    '''
    trajectory = [perturbate(x,road_size, nbr_samples, alpha) for x in solution]
    return np.array(trajectory).reshape(-1,2)


def generate_family_trajectories(solution, nbr, road_size, alpha=2):
    '''Generate a family of trajectories (valid and non valid)
    solution: average optimal solution
    nbr: number of trajectories to generate
    road_size: road size in pixels (allows to estimate covariance matrix)
    
    '''
    family = [perturbate_trajectory(solution, road_size, alpha=alpha) for i in range(nbr)]
    solution = solution.reshape(-1, solution.shape[0], 2)
    solution = np.concatenate((solution, np.array(family)), axis=0)
    return solution


def plot_trajectories(grid, family):
    '''Plot family of solutions of a given maze
    
    '''
    plt.figure(figsize=(10, 10))

    plt.imshow(grid, cmap=plt.cm.binary, 
           interpolation='nearest', 
           origin='lower')

    for i in range(len(family)):
        plt.plot(family[i][:,1], family[i][:,0])
        
    plt.xlim((0,grid.shape[0]-1))
    plt.ylim((0,grid.shape[1]-1))
    plt.axis('off')

    
def plot_solution(grid, solution):
    '''Plot maze an a single solution
    
    '''
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=plt.cm.binary, 
           interpolation='nearest', 
           origin='lower')

    plt.scatter(solution[:, 1], solution[:, 0],s=10 , c='r', marker='+')

    plt.xlim((0,grid.shape[0]-1))
    plt.ylim((0,grid.shape[1]-1))
    plt.axis('off')


def showPNG(grid, axis=False):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest', 
                origin='lower')
    if not axis:
        plt.xticks([]), plt.yticks([])
    plt.show()
    

def sibling_mazes(maze, nbr_trajectories, w, h, alpha=2):
 
    maze_grid = clear_maze_entrances(maze['grid'], maze['start'], maze['end'])
    maze_grid = resize_grid(maze_grid['grid'], (w, h))
     
    # Generate path planning
    path_planning = generate_path_planning(maze['grid'], maze['start'], maze['end'], maze['solutions'])
    path_planning = resize_grid(path_planning['grid'], (w, h))

    # Generate maze's solution / The solution is a sequence with all points (start to end)
    solution = get_solution(maze['solutions'], maze['start'], maze['end'])        
    solution = upsample_trajectory(solution, (w, h), maze['grid'].shape)

    # Estimate agent size in reference to road
    road_size = estimate_road_width(path_planning)

    # Generate family of solutions / concatenate with solutions
    family = generate_family_trajectories(solution, nbr_trajectories, road_size, alpha)

    # Generate siblings maze
    s_mazes = []
    for idx, s in enumerate(family):
        if idx == 0:  # expert trajectory True / always the first solution in the list
            aux = {'grid':maze['grid'], 'start':maze['start'], 
                    'end':maze['end'], 'solutions':maze['solutions'],
                    'upsampled_solution': s, 'expert': True
                    }
        else:   #. Synthetic trajectories
            # Check if collisions exist
            nbr_collisions, total_points = check_trajectory(s, maze_grid, road_size)
            # If collisions exists keep the generated trajectory
            # Else just discard it
            if nbr_collisions != 0:
                aux = {'grid':maze['grid'], 'start':maze['start'], 
                        'end':maze['end'], 'solutions':maze['solutions'],
                        'upsampled_solution': s, 'expert': False
                        }

        s_mazes.append(aux)

    return np.array(s_mazes)


def sibling_mazes_expert(maze, nbr_trajectories, w, h, alpha=15):
    '''Generate family of valid expert trajectories. Alpha=15 produced 1% 
    of rejected trajectories.
    '''
    maze_grid = clear_maze_entrances(maze['grid'], maze['start'], maze['end'])
    maze_grid = resize_grid(maze_grid['grid'], (w, h))
       
    # Generate path planning
    path_planning = generate_path_planning(maze['grid'], maze['start'], maze['end'], maze['solutions'])
    path_planning = resize_grid(path_planning['grid'], (w, h))

    # Generate maze's solution / The solution is a sequence with all points (start to end)
    solution = get_solution(maze['solutions'], maze['start'], maze['end'])        
    solution = upsample_trajectory(solution, (w, h), maze['grid'].shape)

    # Estimate agent size in reference to road
    road_size = estimate_road_width(path_planning)

    # Generate family of solutions / concatenate with solutions
    family = generate_family_trajectories(solution, nbr_trajectories, road_size, alpha)

    # Generate siblings maze
    s_mazes = []
    for idx, s in enumerate(family):
        # Verify if the trajectory is 'expert' or not
        nbr_collisions, total_points = check_trajectory(s, maze_grid, road_size)

        if nbr_collisions == 0:  # expert trajectory True / always the first solution in the list
            aux = {'grid':maze['grid'], 'start':maze['start'], 
                    'end':maze['end'], 'solutions':maze['solutions'],
                    'upsampled_solution': s, 'expert': True
                    }
        # All non-expert trajectories are rejected
        #else:   #. Reject trajectories that are not valid
            #print('rejected trajectory') 
            #aux = {'grid':maze['grid'], 'start':maze['start'], 
            #        'end':maze['end'], 'solutions':maze['solutions'],
            #        'upsampled_solution': s, 'expert': False
            #        }

        s_mazes.append(aux)

    return np.array(s_mazes)


def play_raw_data(grid, solution, canvas_size=(64,64)):
    """Plot raw data.

    """
    road_size = estimate_road_width(grid)
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(1)
    l = int(road_size/2)
    w = int(road_size/2)
    for i, pixel in enumerate(solution): 
        # Get predicted canvas with ego pose
        ego_canvas = predicted_grid(pixel, road_size, canvas_size=canvas_size)
        # grid + canvas
        add = cv2.add(grid.T, ego_canvas)
        
        # Display image
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.imshow(add, cmap=plt.cm.binary)
        plt.axis('off')
        time.sleep(1)

        # Calculate overlap loss and show in it title
        plt.title('Overlap loss function {}'.format(overlap_loss(ego_canvas,grid)))
    plt.close()


def predicted_grid(prediction, road_size, angle=90, canvas_size=(64,64)):
    # Estimate agent size as the half of the road size
    w, l = (int(road_size/2), int(road_size/2))
    # Unpack prediction
    x, y = prediction
    # Draw polygon and project into an empty canvas
    polygon = polygon_from_ego((x,y),angle=angle,l=l,w=w)
    canvas = np.zeros(canvas_size)
    predicted_grid = draw_egopose(canvas, polygon)

    return predicted_grid


def check_trajectory(trajectory, grid, road_size):
    '''Check how many points of a given trajectory
    are not consistent with the environment (collision)
    '''
    collision_count = 0
    for pixel in trajectory:
        # Get predicted canvas
        ego_canvas = predicted_grid(pixel, road_size, angle=90)
        # Check if ego canvas and environment overlaps
        add = cv2.add(grid.T, ego_canvas)
        collisions = np.where(add == 2.)
        if collisions[0].any():  # Overlap detected
            collision_count += 1
            
    return collision_count, len(trajectory)
     

def draw_goals_canvas(grid, solution):
    '''Generates a canvas with start and ending points of a trajectory
    
    '''
    # Create empty canvas (1: non drivable)
    canvas = np.zeros_like(grid)
    road_size = estimate_road_width(grid)
    w, l = (int(road_size/2), int(road_size/2))
    
    # Get start and ending points
    start = solution[0]
    end = solution[-1]
    
    for point in [start, end]:
            polygon = polygon_from_ego(point, angle=90, l=l, w=w)
            cv2.drawContours(canvas, [polygon], 0, (1,0,0), -1)  #  1: agent present (polygon)  

    return canvas


def draw_solution_canvas(grid, solution):
    '''Generates a canvas with a desired trajectory
    
    '''
    # Create empty canvas (1: non drivable)
    canvas = np.zeros_like(grid)
    road_size = estimate_road_width(grid)
    w, l = (int(road_size/2), int(road_size/2))
    
    # Get start and ending points
    for point in solution:
            polygon = polygon_from_ego(point, angle=90, l=l, w=w)
            cv2.drawContours(canvas, [polygon], 0, (1,0,0), -1)  #  1: agent present (polygon)  

    return canvas