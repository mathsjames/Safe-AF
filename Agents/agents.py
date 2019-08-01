import numpy as np

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
        action_probabilities = self.exploration.function(xp, self.games_played)
        return action_probabilities

    def get_expected_rewards(self, epistemic_state):
        return [self.expected_utility[epistemic_state][action] for action in self.actions]

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

class Forgetfull_Agent(Simple_Agent):

    ## Similar to Simple_Agent, but recent observations are given greater weight

    def __init__(self, exploration_scheme, learning_scheme, decision_problem, memory_time_discounting):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states

        self.exploration = exploration_scheme
        self.learning_scheme = learning_scheme

        temp1 = {action:15 for action in self.actions}
        temp2 = {action:1 for action in self.actions}
        self.expected_utility = {es:temp1.copy() for es in self.epistemic_states}
        self.times_action_taken = {es:temp2.copy() for es in self.epistemic_states}

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
                x = min(i/(i+1.0), self.memory_time_discounting)

                exp = self.expected_utility[epistemic_state][action]
                self.expected_utility[epistemic_state][action] = (1-x)*reward + x*exp
                self.times_action_taken[epistemic_state][action] += 1
