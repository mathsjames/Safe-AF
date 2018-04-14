import numpy as np
from decision_problem import Decision_Problem
from agents import *
from training_data import *
import matplotlib.pyplot as plt

## WARNING! The implementation is a bit hacky:
## This decision problem uses a modified version of run, and will have to be patched if the superclass is changed

class Sleeping_Beauty_V1(Decision_Problem):

    # Reward given by bet, should result in SIA/thirder
    # Being a halfer yields almost the same reward however

    def __init__(self):
        self.actions = ["GUESS HEADS", "GUESS TAILS"]

        self.states = ["Heads+Monday", "Tails+Monday", "Tails+Tuesday"]

        self.end_states = ["Heads+Monday", "Tails+Tuesday"]

        self.epistemic_states = ["Awake"]

        self.state_to_epistemic_state_dict = {"Heads+Monday":"Awake",
                                              "Tails+Monday":"Awake",
                                              "Tails+Tuesday":"Awake",
                                              "END":"END"
                                             }

        self.causation_dict = {("Tails+Monday", "GUESS HEADS"):"Tails+Tuesday",
                               ("Tails+Monday", "GUESS TAILS"):"Tails+Tuesday",
                              }

        self.utility_dict = {} # This decision problem uses a modified run function instead

        self.description = "Sleeping Beauty, reward by bet"

    def reset_with(self, agent):
        self.finished = False
        HEADS = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        if HEADS:
            self.state = "Heads+Monday"
        else:
            self.state = "Tails+Monday"

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

                if self.state == "Heads+Monday":
                    reward = action_distribution[0]
                else:
                    reward = action_distribution[1]

                _,_ = self.do(action)

                s = Step(epistemic_state, action, reward)
                utility += reward
                step_history.append(s)

            episode = Episode(utility, step_history)

            if learn:
                agent.learn_from([episode])

            history.append(episode)
            for state in interesting_states:
                distribution_history[state].append(agent.get_action_distribution(state)[0])

        for state in interesting_states:
            plt.plot(distribution_history[state])
            plt.title("Probability of " + agent.actions[0] + " in " + state)
            plt.show()

        return history

class Sleeping_Beauty_V2(Decision_Problem):

    # Reward given by game, should result in SSA/halfer

    def __init__(self):
        self.actions = ["GUESS HEADS", "GUESS TAILS"]

        self.states = ["Heads+Monday", "Tails+Monday", "Tails+Tuesday"]

        self.end_states = ["Heads+Monday", "Tails+Tuesday"]

        self.epistemic_states = ["Awake"]

        self.state_to_epistemic_state_dict = {"Heads+Monday":"Awake",
                                              "Tails+Monday":"Awake",
                                              "Tails+Tuesday":"Awake",
                                              "END":"END"
                                             }

        self.causation_dict = {("Tails+Monday", "GUESS HEADS"):"Tails+Tuesday",
                               ("Tails+Monday", "GUESS TAILS"):"Tails+Tuesday",
                              }

        self.utility_dict = {} # This decision problem uses a modified run function instead

        self.description = "Sleeping Beauty, reward by game"

    def reset_with(self, agent):
        self.finished = False
        HEADS = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        if HEADS:
            self.state = "Heads+Monday"
        else:
            self.state = "Tails+Monday"

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

                if self.state == "Heads+Monday":
                    reward = action_distribution[0]
                elif self.state == "Tails+Tuesday":
                    reward = action_distribution[1]
                else:
                    reward = 0

                _,_ = self.do(action)

                s = Step(epistemic_state, action, reward)
                utility += reward
                step_history.append(s)

            episode = Episode(utility, step_history)

            if learn:
                agent.learn_from([episode])

            history.append(episode)
            for state in interesting_states:
                distribution_history[state].append(agent.get_action_distribution(state)[0])

        for state in interesting_states:
            plt.plot(distribution_history[state])
            plt.title("Probability of " + agent.actions[0] + " in " + state)
            plt.show()

        return history
