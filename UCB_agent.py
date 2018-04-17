from exploration_schemes import *

class UCB_Agent:

    # Chooses the action a with the highest value of E(U|a)+c*STD(U|a)

    def __init__(self, learning_scheme, decision_problem, c, prior_mean, prior_std):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states
        self.learning_scheme = learning_scheme

        self.c = c

        temp = {action:prior_mean for action in self.actions}
        self.expected_utility = {es:temp.copy() for es in self.epistemic_states}

        temp = {action:prior_mean**2 for action in self.actions}
        self.mean_squared = {es:temp.copy() for es in self.epistemic_states}

        temp = {action:1 for action in self.actions}
        self.times_action_taken = {es:temp.copy() for es in self.epistemic_states}

        self.total_utility = 0
        self.games_played = 1

    def get_action_distribution(self, epistemic_state):

        mean_values = [self.expected_utility[epistemic_state][action] for action in self.actions]
        mean_sq_values = [self.mean_squared[epistemic_state][action] for action in self.actions]
        std_values = [s-m**2 for (m,s) in zip(mean_values, mean_sq_values)]
        action_potential = [m+self.c*s for (m,s) in zip(mean_values, std_values)]
        best = max(action_potential)
        action_probabilities = [1 if x==best else 0 for x in action_potential]
        action_probabilities = [x/sum(action_probabilities) for x in action_probabilities]

        return action_probabilities

    def get_expected_rewards(self, epistemic_state):
        return self.expected_utility[epistemic_state]

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
                mean_sq = self.mean_squared[epistemic_state][action]

                self.expected_utility[epistemic_state][action] = (reward+exp*i)/(i+1.0)
                self.mean_squared[epistemic_state][action] = (reward**2+mean_sq*i)/(i+1.0)
                self.times_action_taken[epistemic_state][action] += 1


class Softmax_UCB_Agent(UCB_Agent):

    def __init__(self, learning_scheme, decision_problem, c, prior_mean, prior_std, temperature):
        UCB_Agent.__init__(self, learning_scheme, decision_problem, c, prior_mean, prior_std)
        self.softmax = Softmax(temperature)

    def get_action_distribution(self, epistemic_state):

        mean_values = [self.expected_utility[epistemic_state][action] for action in self.actions]
        mean_sq_values = [self.mean_squared[epistemic_state][action] for action in self.actions]
        std_values = [s-m**2 for (m,s) in zip(mean_values, mean_sq_values)]
        action_potential = [m+self.c*s for (m,s) in zip(mean_values, std_values)]
        action_probabilities = self.softmax.function(action_potential)

        return action_probabilities
