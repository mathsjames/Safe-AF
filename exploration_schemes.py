import numpy as np

class Epsilon_Greedy:

    def __init__(self, epsilon):
        self.epsilon = float(epsilon)

    def function(self, x):
        exploring = np.random.choice([True, False], 1, p=[self.epsilon, 1-self.epsilon])[0]

        if exploring:
            return [1.0/len(x) for i in x]
        else:
            mx = max(x)
            mxs = list(filter(lambda x: x == mx, x))
            return [1.0/len(mxs) if i==mx else 0 for i in x]

# Epsilon-First?
# Epsilon Decreasing?

class Softmax:

    def __init__(self, temperature):
        self.temperature = float(temperature)

    def function(self, x):
        x = [i/self.temperature for i in x]
        e_x = np.exp(x-np.max(x))
        return e_x / e_x.sum(axis=0)

class More_Advanced_Softmax:

    def __init__(self, cooling_function):
        self.cooling_function = cooling_function
        #Cooling function should be a function from number of games played to temperature

    def function(self, x, games_played):
        temperature = self.cooling_function.function(games_played)
        x = [i/temperature for i in x]
        e_x = np.exp(x-np.max(x))
        return e_x / e_x.sum(axis=0)

class exponential_cooling:
    def function(self, games_played):
        if  100*(0.99**games_played) > 0.000001:
            temperature = 100*(0.99**games_played)
        else:
            temperature = 0.000001
        return temperature


class UCB_exploration_scheme:

    def __init__(self, c):
        self.c = c

    def function(self, exp, std):

        return 0

    def choose(self, agent):
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)
