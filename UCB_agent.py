from exploration_schemes import *
import math

class UCB_Agent:

    # UCB1

    def __init__(self, learning_scheme, decision_problem, prior):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states
        self.learning_scheme = learning_scheme

        temp = {action:prior for action in self.actions}
        self.expected_utility = {es:temp.copy() for es in self.epistemic_states}

        temp = {action:1 for action in self.actions}
        self.times_action_taken = {es:temp.copy() for es in self.epistemic_states}

        self.total_utility = 0
        self.games_played = 1

    def get_action_distribution(self, epistemic_state):

        es = epistemic_state
        exp = {a:self.expected_utility[es][a] for a in self.actions}
        T = sum([self.times_action_taken[es][a] for a in self.actions])
        c = 2*math.log(T)**0.5 # derived from Hoeffding's inequality
        action_potential = [exp[a]+c/self.times_action_taken[es][a] for a in self.actions]

        best = max(action_potential)
        action_probabilities = [1 if x==best else 0 for x in action_potential]
        action_probabilities = [x/sum(action_probabilities) for x in action_probabilities]

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


class Softmax_UCB_Agent(UCB_Agent):

    # Softmaxes action potential instead of choosing greedily

    def __init__(self, learning_scheme, decision_problem, prior, temperature):
        UCB_Agent.__init__(self, learning_scheme, decision_problem, prior)
        self.softmax = Softmax(temperature)

    def get_action_distribution(self, epistemic_state):

        es = epistemic_state
        exp = {a:self.expected_utility[es][a] for a in self.actions}
        T = sum([self.times_action_taken[es][a] for a in self.actions])
        c = 2*math.log(T)**0.5 # derived from Hoeffding's inequality
        action_potential = [exp[a]+c/self.times_action_taken[es][a] for a in self.actions]

        action_probabilities = self.softmax.function(action_potential)

        return action_probabilities
