import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 1000

STATS_EVERY = 100
bins = 30
DISCRETE_OS_SIZE = [bins, bins] #* len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 1 #measure of how much exploration we want to do
# epsilon serves to add randomndess otherwise once the agent gets to the finishline it is never
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 #TO INTEGER SO we don't have a float
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low =2, high = 0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min': [], 'max': []} #dictionrary to track ep number, average every window, min (worst model we had) and max (best model)
# average may going up. might have cases in which  we want min not so bad than a high average

def get_descrete_state(state):
    discrete_state=(state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward =0
    discrete_state = get_descrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_descrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )] #exact q value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE* (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q #updating after we took the step
        elif new_state[0] >= env.goal_position: #car gets to the flag :)
            print(f"Flag reached at episode {episode}")
            q_table[discrete_state + (action, )] = 0 #reward for completing

        discrete_state = new_discrete_state


    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/ len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    #print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])}")
env.close()

'''
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'], label = "avg")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'], label = "min")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'], label = "max")
plt.legend(loc=1)
plt.show()
'''

def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    #q_actions = np.argmax(q_table, axis=2)  # best action for each state
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(q_image, cmap='jet');
    cbar = fig.colorbar(cax)
    '''
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    '''
    ax.grid(False)
    ax.set_title(f"Q-table, size: {(q_table.shape)} \n Value for discount rate of {DISCOUNT} and \n learning rate of {LEARNING_RATE} ")
    #plt.title()
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    cbar.set_label('Value')
    plt.show()

plot_q_table(q_table)
