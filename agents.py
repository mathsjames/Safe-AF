import numpy as np
from exploration

## Exploration Schemes ##



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
            average_reward = sum([i.reward for i in episode.steps])/len(episode.steps)
            e = episode.copy()
            for step in e.steps:
                step.reward = average_reward
            ans.append(e)
        return ans

class Total:
    # Q-learning
    def __init__(self):
        pass
    def process(self, history):
        ans = []
        for episode in history:
            total = 0
            e = episode.copy()
            for step in e.steps[::-1]:
                total += step.reward
                step.reward = total
            ans.append(e)
        return ans

## Agents ##

class Simple_Agent:

    def __init__(self, exploration_scheme, learning_scheme, decision_problem, prior):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states

        self.exploration = exploration_scheme
        self.learning_scheme = learning_scheme

        temp = {action:prior for action in self.actions}
        self.expected_utility = {es:temp.copy() for es in self.epistemic_states}

        temp = {action:1 for action in self.actions}
        self.times_action_taken = {es:temp.copy() for es in self.epistemic_states}

        self.total_utility = 0
        self.games_played = 1

    def get_action_distribution(self, epistemic_state):

        xp = [self.expected_utility[epistemic_state][action] for action in self.actions]
        action_probabilities = self.exploration.function(xp)
        return action_probabilities

    def learn_from(self, training_data):

        training_data = self.learning_scheme.process(training_data)

        for episode in training_data:

            self.games_played += 1
            self.total_utility += episode.utility

            for step in episode.steps:

                epistemic_state = step.epistemic_state
                action = step.action
                reward = step.reward

                i = self.times_action_taken[epistemic_state][action]
                exp = self.expected_utility[epistemic_state][action]
                self.expected_utility[epistemic_state][action] = (reward+exp*i)/(i+1.0)
                self.times_action_taken[epistemic_state][action] += 1

class More_Advanced_Agent(Simple_Agent):

    def get_action_distribution(self, epistemic_state):
        xp = [self.expected_utility[epistemic_state][action] for action in self.actions]
        action_probabilities = self.exploration.function(xp, self.games_played)
        return action_probabilities

class Forgetfull_Agent(Simple_Agent):

    def __init__(self, exploration_scheme, learning_scheme, decision_problem, memory_time_discounting):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states

        self.exploration = exploration_scheme
        self.learning_scheme = learning_scheme

        temp = {action:15 for action in self.actions}
        self.expected_utility = {es:temp.copy() for es in self.epistemic_states}
        self.times_action_taken = {es:temp.copy() for es in self.epistemic_states}

        self.total_utility = 0
        self.games_played = 1

        self.memory_time_discounting = memory_time_discounting

    def learn_from(self, training_data):

        training_data = self.learning_scheme.process(training_data)

        for episode in training_data:

            self.games_played += 1
            self.total_utility += episode.utility

            for step in episode.steps:

                epistemic_state = step.epistemic_state
                action = step.action
                reward = step.reward

                i = self.times_action_taken[epistemic_state][action]
                exp = self.expected_utility[epistemic_state][action]
                self.expected_utility[epistemic_state][action] = (1-self.memory_time_discounting)*reward + self.memory_time_discounting*exp
                self.times_action_taken[epistemic_state][action] += 1
