import numpy as np

class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5, epsilon=1.3, min_epsilon=0.0, decay_rate=0.95):
        self.__num_arms = num_arms #private
        ## IMPLEMENTATION

        self.__num_arms = num_arms
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        # Initialize estimated Q-values and action counts for each arm.
        self.Q_values = np.zeros(self.__num_arms, dtype=float)
        self.action_counts = np.zeros(self.__num_arms, dtype=int)
    
    def update_state(self, action, reward):
        """
            TODO:
            Based on your choice of algorithm, use the the current action and
            reward to update the state of the agent.
            Optinal function, only use if needed.
        """
        ## IMPLEMENTATION
        # Increment the count for the chosen action.
        self.action_counts[action] += 1
        
        # Update the Q-value using an incremental update rule.
        old_Q = self.Q_values[action]
        self.Q_values[action] = old_Q + (1.0 / self.action_counts[action]) * (reward - old_Q)
        
        # Decay epsilon multiplicatively, ensuring it doesn't fall below min_epsilon.
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
    
    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        ## IMPLEMENTATION
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.__num_arms)
        else:
            return np.argmax(self.Q_values)
