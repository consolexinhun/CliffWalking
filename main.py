import numpy as np
import matplotlib.pyplot as plt
world_width = 12
world_height = 4

epsilon = 0.1
alpha = 0.1

gamma = 1

start = [world_height-1, 0]
goal = [world_height-1, world_width-1]

action_up = 0
action_down = 1
action_left = 2
action_right = 3
actions = [0, 1, 2, 3]


def Q_learning():
    episodes = 500
    runs = 100
    rewards_q_learning = np.zeros(episodes)
    q_learning_table = np.zeros((world_height, world_width, 4))
    for r in range(runs):
        for episode in range(episodes):
            state = start
            rewards = 0
            while state != goal:
                if np.random.binomial(1, epsilon) == 1:
                    action = np.random.choice(actions)
                else:
                    values = q_learning_table[state[0], state[1], :]
                    action = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
                i, j = state
                if action == action_up:  # up
                    next_state = [max(i - 1, 0), j]
                elif action == action_down:
                    next_state = [min(i + 1, world_height - 1), j]
                elif action == action_left:
                    next_state = [i, max(j - 1, 0)]
                elif action == action_right:
                    next_state = [i, min(j + 1, world_width - 1)]
                else:
                    assert False

                reward = -1
                if (action == action_down and i == world_height - 2 and 1 <= j <= world_width - 2) or (
                        action == action_right and state == start):
                    reward = -100
                    next_state = start
                rewards += reward

                q_learning_table[state[0], state[1], action] += alpha * (reward + gamma * np.max(q_learning_table[next_state[0], next_state[1], :]) - q_learning_table[state[0], state[1], action])
                state = next_state
            rewards_q_learning[episode] += rewards
    rewards_q_learning /= runs

    plt.plot(rewards_q_learning, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.legend()
    plt.ylim([-100, 0])

    optimal_policy = np.zeros((world_height, world_width), dtype=object)
    for i in range(world_height):
        for j in range(world_width):
            if [i, j] == goal:
                optimal_policy[i][j] = 'G'
                continue
            bestAction = np.argmax(q_learning_table[i, j, actions])
            if bestAction == action_right:
                optimal_policy[i][j] = '→'
            elif bestAction == action_left:
                optimal_policy[i][j] = "←"
            elif bestAction == action_up:
                optimal_policy[i][j] = "↑"
            elif bestAction == action_down:
                optimal_policy[i][j] = "↓"
    print("-----------Optimal Q-learning-------------")
    for row in optimal_policy:
        print(row)

def Sarsa():
    episodes = 500
    runs = 100
    rewards_q_learning = np.zeros(episodes)
    q_learning_table = np.zeros((world_height, world_width, 4))
    for r in range(runs):
        for episode in range(episodes):
            state = start # 左下角的数，包含两个值
            rewards = 0
            if np.random.binomial(1, epsilon) == 1:
                action = np.random.choice(actions)
            else:
                values = q_learning_table[state[0], state[1], :]
                action = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
            while state != goal:
                i, j = state
                if action == action_up:  # up
                    next_state = [max(i - 1, 0), j]
                elif action == action_down:
                    next_state = [min(i + 1, world_height - 1), j]
                elif action == action_left:
                    next_state = [i, max(j - 1, 0)]
                elif action == action_right:
                    next_state = [i, min(j + 1, world_width - 1)]
                else:
                    assert False

                reward = -1
                if (action == action_down and i == world_height - 2 and 1 <= j <= world_width - 2) or (
                        action == action_right and state == start):
                    reward = -100
                    next_state = start
                rewards += reward

                if np.random.binomial(1, epsilon) == 1:
                    next_action = np.random.choice(actions)
                    q_learning_table[state[0], state[1], action] += alpha * (reward + gamma * (q_learning_table[next_state[0], next_state[1], next_action]) -q_learning_table[state[0], state[1], action])
                else:
                    values = q_learning_table[next_state[0], next_state[1], :]
                    next_action = np.random.choice([action for action, value in enumerate(values) if value == np.max(values) ])
                    # q_learning_table[state[0], state[1], action] += alpha * (reward + gamma * np.max(q_learning_table[next_state[0], next_state[1], :]) - q_learning_table[state[0], state[1], action])
                    q_learning_table[state[0], state[1], action] += alpha * (reward + gamma * q_learning_table[next_state[0], next_state[1], next_action] - q_learning_table[state[0], state[1], action])
                state = next_state
                action = next_action
            rewards_q_learning[episode] += rewards
    rewards_q_learning /= runs
    plt.plot(rewards_q_learning, label="Sarsa")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.legend()
    plt.ylim([-100, 0])

    optimal_policy = np.zeros((world_height, world_width), dtype=object)
    for i in range(world_height):
        for j in range(world_width):
            if [i, j ] == goal:
                optimal_policy[i][j] = 'G'
                continue
            bestAction = np.argmax(q_learning_table[i, j, actions])
            if bestAction == action_right:
                optimal_policy[i][j] = '→'
            elif bestAction == action_left:
                optimal_policy[i][j] = "←"
            elif bestAction == action_up:
                optimal_policy[i][j] = "↑"
            elif bestAction == action_down:
                optimal_policy[i][j] = "↓"

    print("-----------Optimal Sarsa-------------")
    for row in optimal_policy:
        print(row)


def ExpectedSarsa():
    episodes = 500
    runs = 100
    rewards_q_learning = np.zeros(episodes)
    q_learning_table = np.zeros((world_height, world_width, 4))
    for r in range(runs):
        for episode in range(episodes):
            state = start # 左下角的数，包含两个值
            rewards = 0
            while state != goal:
                if np.random.binomial(1, epsilon) == 1:
                    action = np.random.choice(actions)
                else:
                    values = q_learning_table[state[0], state[1], :]
                    action = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
                i, j = state
                if action == action_up:  # up
                    next_state = [max(i - 1, 0), j]
                elif action == action_down:
                    next_state = [min(i + 1, world_height - 1), j]
                elif action == action_left:
                    next_state = [i, max(j - 1, 0)]
                elif action == action_right:
                    next_state = [i, min(j + 1, world_width - 1)]
                else:
                    assert False

                reward = -1
                if (action == action_down and i == world_height - 2 and 1 <= j <= world_width - 2) or (
                        action == action_right and state == start):
                    reward = -100
                    next_state = start
                rewards += reward

                q_learning_table[state[0], state[1], action] += alpha * (reward + gamma * np.mean(q_learning_table[next_state[0], next_state[1], :]) - q_learning_table[state[0], state[1], action])
                state = next_state

            rewards_q_learning[episode] += rewards
    rewards_q_learning /= runs
    plt.plot(rewards_q_learning, label="ExpectedSarsa")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.legend()
    plt.ylim([-100, 0])

    optimal_policy = np.zeros((world_height, world_width), dtype=object)
    for i in range(world_height):
        for j in range(world_width):
            if [i, j ] == goal:
                optimal_policy[i][j] = 'G'
                continue
            bestAction = np.argmax(q_learning_table[i, j, actions])
            if bestAction == action_right:
                optimal_policy[i][j] = '→'
            elif bestAction == action_left:
                optimal_policy[i][j] = "←"
            elif bestAction == action_up:
                optimal_policy[i][j] = "↑"
            elif bestAction == action_down:
                optimal_policy[i][j] = "↓"

    print("-----------Optimal Expected  Sarsa-------------")
    for row in optimal_policy:
        print(row)

plt.figure()
Q_learning()
Sarsa()
# ExpectedSarsa()
plt.show()


plt.figure()
Q_learning()
plt.show()

plt.figure()
Sarsa()
plt.show()


# plt.figure()
# ExpectedSarsa()
# plt.show()



