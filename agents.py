import numpy as np

## Exploration Schemes ##

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
        temperature = cooling_function(games_played)
        x = [i/temperature for i in x]
        e_x = np.exp(x-np.max(x))
        return e_x / e_x.sum(axis=0)


def exponential_cooling(games_played):
    # Example of cooling function
    temperature = 1000*(0.999**games_played)
    return temperature




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

## Training data preprocessing functions ##

class Identity_Function: # Should be used for all 1-step games
    def __init__(self):
        pass
    def process(self, history):
        return history

class Temporal_Discounting:
    def __init__(self, rate):
        self.rate = rate
    def process(self, history):
        # TODO
        return history

class Average:
    def __init__(self):
        pass
    def process(self, history):
        ans = []
        for episode in history:
            average_utility = sum([i[2] for i in episode])/len(episode)
            ans.append([(epistemic_state, action, average_utility) for (epistemic_state, action, _) in episode])
        return ans

## Agents ##

class Simple_Agent:

    def __init__(self, exploration_scheme, learning_scheme, decision_problem):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states

        self.exploration = exploration_scheme
        self.learning_scheme = learning_scheme

        temp = {action:1 for action in self.actions}
        self.expected_utility = {es:temp.copy() for es in self.epistemic_states}
        self.times_action_taken = {es:temp.copy() for es in self.epistemic_states}

        self.total_utility = 0
        self.games_played = 0

    def get_action_distribution(self, epistemic_state):

        xp = [self.expected_utility[epistemic_state][action] for action in self.actions]
        action_probabilities = self.exploration.function(xp)
        return action_probabilities

    def learn_from(self, training_data):

        training_data = self.learning_scheme.process(training_data)

        for episode in training_data:

            self.games_played += 1

            for epistemic_state, action, utility in episode:

                self.total_utility += utility

                i = self.times_action_taken[epistemic_state][action]
                exp = self.expected_utility[epistemic_state][action]
                self.expected_utility[epistemic_state][action] = (utility+exp*i)/(i+1.0)
                self.times_action_taken[epistemic_state][action] += 1


class More_Advanced_Agent(Simple_Agent):

    def get_action_distribution(self, epistemic_state):
        xp = [self.expected_utility[epistemic_state][action] for action in self.actions]
        action_probabilities = self.exploration.function(xp, self.games_played)
        return action_probabilities
