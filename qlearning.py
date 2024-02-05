import datetime
import os
import numpy as np
import imageio
import random
from experience import MazeExperience
from maze import visualize_maze

step_number = 0

def q_learning_train(maze, rat, model, **options):
    global step_number
    epsilon = 0.1


    max_memory = options.get('max_memory', 1000)
    num_epochs = options.get('num_epochs', 15000)
    data_size = options.get('data_size', 50)
    model_name = options.get('model_name', 'model')
    weights_file = options.get('weights_file', "")
    start_time = datetime.datetime.now() 




    if weights_file:
        print(f"Loading weights from file: {weights_file}")
        model.load_weights(weights_file)


    if not os.path.exists('maze_snapshots'):
        os.makedirs('maze_snapshots')  # Creates a new directory named 'maze_snapshots' if undefined

    experience = MazeExperience(model, maze, max_memory=100, discount=0.95)  # Create a MazeExperience object


    win_history = []
    history_size = 50
    win_rate = 0.0
    shortest_path_maze = None
    counter = 0


    for epoch in range(num_epochs):
        loss = 0.0
        rat_cell = (0,0)
        rat.reset()
        maze.reset(rat_cell)
        game_over = False
        step_number += 1

        num_episodes = 0

        while not game_over:
            env_state = maze.observe()
            # print("Environment State:", env_state)  # Add this print statement
            valid_actions = maze.valid_actions()  # Moves the rat


            if not valid_actions:
                break

            prev_env_state = env_state


            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(model.predict(prev_env_state))  # Use the predict method


            env_state, reward, game_status = maze.act(action)
            
            # visualize_maze(maze, step_number)  # Save maze snapshot

            if game_status in ['win', 'lose']:
                win_history.append(game_status == 'win')
                game_over = True
            else:
                game_over = False


            episode = [prev_env_state, action, reward, env_state, game_over]
            experience.remember(episode)
            num_episodes += 1

            state = prev_env_state
            next_state = env_state
            done = (game_status != 'not_over')
            discount_factor = 0.9


            # Calculate TD error using the model
            action_index = action  # Replace this with your action index
            loss = model.td_error(state, action_index, reward, next_state, done, discount_factor)
            model.history.add_loss(loss)


            inputs, targets = experience.get_data(data_size=10)
            history = model.fit(inputs, targets, epochs=16, batch_size=16)

            if game_over:
                visualize_maze(maze, step_number)  # Save maze snapshot

        array = maze.maze
        count = np.count_nonzero(array == 0.5)

        if count > 0:
            if count < counter or counter == 0:
                counter = count
                shortest_path_maze = maze


        win_rate = sum(win_history[-history_size:]) / history_size


        dt = datetime.datetime.now() - start_time # Time elapsed since start of training
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | Time: {}"
        print(template.format(epoch, num_epochs-1, loss, num_episodes, sum(win_history), win_rate, t))


        if win_rate > 0.9:
            epsilon = 0.05 # Reduces the exploration factor; uses the model 95% approx.

        if win_rate ==  1.0:
            break


    model.save_weights('saved_weights.npz')  #Saving weights

    visualize_maze(shortest_path_maze, step_number+1)

    images = []
    for filename in sorted(os.listdir('maze_snapshots')):
        images.append(imageio.imread(f'maze_snapshots/{filename}'))


    imageio.mimsave('maze_solution.gif', images, duration=0.5)  # Create GIF


    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print(f"Epochs: {epoch+1}, Max Memory: {max_memory}, Data Size: {data_size}, Time: {t}")

    return seconds

def format_time(seconds):
    if seconds < 400:
        return f"{seconds:.1f} seconds"
    elif seconds < 4000:
        return f"{seconds / 60:.1f} minutes"
    else:
        return f"{seconds / 3600:.1f} hours"