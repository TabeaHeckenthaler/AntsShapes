import numpy as np
import networkx as nx

# Define the network using NetworkX
network = nx.Graph()
network.add_nodes_from([0, 1, 2, 3])
network.add_edges_from([(0, 1), (1, 2), (2, 3)])

# Define the discount factor
gamma = 0.8

# Define the temperature
temperature = 0.8

# Define the reward function as a dictionary
reward_table = {
    (0, 1): 0,
    (1, 2): 1,
    (2, 3): 2
}

# Initialize the Q-table with zeros
num_states = len(network.nodes)
num_actions = len(network.edges)
q_table = np.zeros((num_states, num_actions))

possible = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0],
                     [0, 0, 1]]).astype(bool)

# Perform Q-learning
epochs = 100
for epoch in range(epochs):
    state = 0  # Starting state
    print(epoch)
    while state != 3:  # Terminate when reaching the goal state
        # Select an action based on the Q-table and temperature
        probabilities = np.exp(q_table[state][possible[state]] / temperature) / \
                        np.sum(np.exp(q_table[state][possible[state]] / temperature))
        action = np.random.choice(range(len(probabilities)), p=probabilities)

        # Get the next state based on the selected action
        next_state = list(network.neighbors(state))[action]

        # Get the reward for the state-action pair
        reward = reward_table.get((state, next_state), 0)

        # Update the Q-table
        q_table[state, action] = reward + gamma * np.max(q_table[next_state])

        # Transition to the next state
        state = next_state
    print('wone')

# Print the learned Q-table
print("Learned Q-table:")
print(q_table)
