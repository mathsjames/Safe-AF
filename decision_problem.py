import numpy as np
from training_data import *
import matplotlib.pyplot as plt

class Decision_Problem:

    def __init__(self):
        self.actions = []
        self.states = []
        self.end_states = []
        self.epistemic_states = []

        self.state_to_epistemic_state_dict = {}

        self.causation_dict = {}

        self.utility_dict = {}

        self.description("")

    def reset_with(self, agent):
        self.finished = False
        self.state = None

    #runs and lets the agent learn after each instance of the problem if input variable learn is True.
    def run(self, agent, iterations, learn=False, interesting_states=[]):
        history = []

        distribution_history = {state:[] for state in interesting_states}

        for i in range(iterations):

            self.reset_with(agent)
            step_history = []
            utility = 0

            while not self.finished:

                epistemic_state = self.epistemic_state()
                action_distribution = agent.get_action_distribution(epistemic_state)
                action = np.random.choice(self.actions, 1, p=action_distribution)[0]
                _, reward = self.do(action)

                s = Step(epistemic_state, action, reward)
                utility += reward
                step_history.append(s)

            episode = Episode(utility, step_history)

            if learn:
                agent.learn_from([episode])

            history.append(episode)
            for state in interesting_states:
                #distribution_history[state].append(agent.get_action_distribution(state)[0])
                distribution_history[state].append(agent.expected_utility[state][agent.actions[1]])

        for state in interesting_states:
            plt.plot(distribution_history[state])
            plt.title("EXP of " + agent.actions[0] + " in " + state)
            plt.show()

        return history

    def do(self, action):

        self.state = self.cause(self.state, action)
        utility = self.utility(self.state)

        if self.state in self.end_states:
            self.finished = True

        return (self.epistemic_state(), utility)

    def utility(self, state):
        try:
            return self.utility_dict[state]
        except:
            return 0

    def epistemic_state(self):
        return self.state_to_epistemic_state_dict[self.state]

    def cause(self, state, action):
        try:
            return self.causation_dict[(state, action)]
        except:
            return state
