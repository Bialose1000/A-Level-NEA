from maze import Maze
from rat import Rat
import numpy as np
from menu import InteractiveMaze
from neuralnetwork import NeuralNetwork
from qlearning import q_learning_train
from menu import create_image_viewer

def MachineLearning(size):
    maze_size = size
    maze_layout = np.ones((maze_size, maze_size))
    interactive_maze = InteractiveMaze(maze_layout)
    interactive_maze.show()
    updated_maze_layout = interactive_maze.maze

    rat = Rat((0,0), maze_size)
    maze = Maze(updated_maze_layout, rat)

    input_size = len(maze_layout) * len(maze_layout[0])  # Flatten the maze layout
    num_actions = 4 
    model = NeuralNetwork(input_size, num_actions)

    q_learning_train(maze, rat, model, max_memory=100, num_epochs=150, data_size=50)
    create_image_viewer('C:/Users/DELL/OneDrive/Documents/A Level/maze_snapshots')
