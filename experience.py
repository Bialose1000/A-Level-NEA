import random
import numpy as np

class Experience:
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = []
        self.num_actions = model.num_actions


    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]


    def predict(self, envstate):
        return self.model.predict(envstate)[0]


    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)


        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))


        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate


            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


    def get_path(self):
        return self.path


class MazeExperience(Experience):
    def __init__(self, model, qmaze, max_memory=100, discount=0.95):
        super().__init__(model, max_memory, discount)
        self.qmaze = qmaze

    def observe(self):
        return self.qmaze.observe() # Takes snapshot of the current state of the maze

    def act(self, action):
        envstate, reward, status = self.qmaze.act(action)
        return envstate, reward, status

    def remember_qmaze(self, num_steps=10):
        initial_state = self.qmaze.observe()

        for _ in range(num_steps):
            action = random.choice(self.qmaze.valid_actions())
            envstate, reward, status = self.act(action)
            episode = [initial_state, action, reward, envstate, (status != 'not_over')]
            self.remember(episode)
