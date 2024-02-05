LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3


class Rat:
    def __init__(self, initial_position, maze_size):
        self.initial_position = initial_position
        self.position = initial_position
        self.maze_size = maze_size  

    def move(self, action):
        row, col = self.get_position()
        if action == 1 and row > 0:
            row -= 1
        elif action == 3 and row < self.maze_size - 1:
            row += 1
        elif action == 0 and col > 0:
            col -= 1
        elif action == 2 and col < self.maze_size - 1:
            col += 1
        self.set_position((row, col))

    def get_position(self):
        return self.position
    
    def set_position(self, position):
        self.position = position

    def reset(self):
        self.position = self.initial_position