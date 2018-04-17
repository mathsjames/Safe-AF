import math

class EXP3_Agent:

    def __init__(self, learning_scheme, decision_problem, epsilon, prior):

        self.actions = decision_problem.actions
        self.epistemic_states = decision_problem.epistemic_states
        self.learning_scheme = learning_scheme

        self.epsilon = epsilon

        temp = {action:prior for action in self.actions}
        self.expected_utility = {es:temp.copy() for es in self.epistemic_states}

        temp = {action:1 for action in self.actions}
        self.weights = {es:temp.copy() for es in self.epistemic_states}

        temp = {action:1 for action in self.actions}
        self.times_action_taken = {es:temp.copy() for es in self.epistemic_states}

        self.total_utility = 0
        self.games_played = 1

    def get_action_distribution(self, epistemic_state):

        sum_of_weights = sum([self.weights[epistemic_state][action] for action in self.actions])
        normalised_weights = {action:(self.weights[epistemic_state][action]/sum_of_weights) for action in self.actions}
        action_probabilities = [(1-self.epsilon)*normalised_weights[action] + self.epsilon*(1.0/len(self.actions)) for action in self.actions]

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

                action_distribution = self.get_action_distribution(epistemic_state)
                action_probability = action_distribution[self.actions.index(action)]

                er = float(reward/action_probability) # "estimated reward"
                K = len(self.actions)
                w = self.weights[epistemic_state][action]
                c = self.epsilon

                self.weights[epistemic_state][action] = w*math.exp(c*er/K)

                # necessary to prevent weights from growing out of control, should not affect the output of the algorithm
                for ep in self.epistemic_states:
                    sum_of_weights = sum([self.weights[epistemic_state][action] for action in self.actions])
                    self.weights[ep] = {action:self.weights[ep][action]/sum_of_weights for action in self.actions}

                i = self.times_action_taken[epistemic_state][action]
                exp = self.expected_utility[epistemic_state][action]

                self.expected_utility[epistemic_state][action] = (reward+exp*i)/(i+1.0)
                self.times_action_taken[epistemic_state][action] += 1
