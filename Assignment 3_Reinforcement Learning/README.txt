The folder contains to python scripts, a main version and a working one:


##########################   Qlearning.py ##########################

Run the Qlearning.py script with python 3. It includes a loop performing an
epsilon-greedy Q learning algorithm on the Mountain Car Gym environment.

Default parameters:
 - LEARNING_RATE = 0.1
 - DISCOUNT = 0.95
 - epsilon = 1 (which is going to be decayed)
 - EPISODES = 25000

The script includes the following Functions:
get_descrete_state() which discretizes the continuous state-space
   - input = state
   - default bin size is 30

plot_q_table() which plots an heatmap for the q table
  - input = q_table
  The function can also include the action corresponding to the highest state-value function
  however, in the main version of the script those have been commented out.


##########################   Optimal Q Learning.ipynb   ########################## 

This notebook differs from Qlearning.py in that the Q learning process is turned into a function that receieves a learning rate and discount factors as input parameters. The second cell performs grid search using different user specified values for the learning rate and discount factor, appends the evolution of the rewards through the episodes and computes the average the minimum rewards, the average of the maximum rewards and the mean of the average rewards for every parameter combination, these are stored in a dataframe, which is sorted by mean average reward. Lastly, the best performing combination of parameters is selected and the value function for them is plotted
