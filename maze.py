import numpy as np
import matplotlib.pyplot as plt
from rat import Rat

rat_mark = 0.5
LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

class Maze:
    def __init__(self, maze, rat):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        if self.target in self.free_cells:
            self.free_cells.remove(self.target)
        else:
            raise Exception("Target cell is not in the list of free cells")
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell is blocked!")
        if rat.get_position() not in self.free_cells:
            raise Exception("Invalid Rat Location: must be in a free cell")
        self.rat = rat
        self.reset(rat.get_position())

    def reset(self, rat):
        self.rat = Rat(rat, self._maze.shape[0])  # Reset the rat's position
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape   
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        rat_row, rat_col = self.rat.get_position()

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))

        valid_actions = self.valid_actions()

        if not valid_actions:
            mode = 'blocked'
        elif action in valid_actions:
            mode = 'valid'
            self.rat.move(action)  # Using the move method of the Rat class
            rat_row, rat_col = self.rat.get_position()  # Updating the rat's position
            self.maze[rat_row, rat_col] = rat_mark  # Mark the rat's new position in the maze

        else:
            mode = 'invalid'

        self.state = (rat_row, rat_col, mode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:  # Rat has reached the target
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.35
        if mode == 'invalid':
            return -0.70
        if mode == 'valid':
            return -0.05

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        game_status = self.game_status()
        envstate = self.observe()
        return envstate, reward, game_status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1)) # Reshapes the maze so that it is only one column long
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0 and (r, c) != self.rat.get_position():
                    canvas[r, c] = 1.0
        row, col = self.rat.get_position()
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, _ = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 'win'
        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, _ = self.state
        else:
            row, col = cell
        actions = [LEFT, UP, RIGHT, DOWN]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(UP)
        elif row == nrows - 1:
            actions.remove(DOWN)
        if col == 0:
            actions.remove(LEFT)
        elif col == ncols - 1:
            actions.remove(RIGHT)
        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(UP)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(DOWN)
        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(LEFT)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(RIGHT)
        return actions

    def display_maze(self):
        canvas = self.draw_env()
        nrows, ncols = self.maze.shape
        start_row, start_col = self.rat.get_position()  # Use get_position method
        target_row, target_col = self.target
        plt.scatter(start_col + 0.5, start_row + 0.5, color='blue', marker='o', s=200)
        plt.scatter(target_row + 0.5, target_col + 0.5, marker='*', s=100, color='red')
        plt.title('Maze with Rat and Target')
        plt.show()


def visualize_maze(maze_obj, step_number):
#   maze_obj.display_maze() # Displays maze
    plt.imshow(maze_obj.maze, cmap='viridis_r')  # Plot the maze
    plt.axis('off')  # Remove the axes
    plt.savefig(f"maze_snapshots/step_{step_number}.png")  # Save the figure
    plt.close()  # Close the figure
