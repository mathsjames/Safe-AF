import numpy as np
from decision_problem import Decision_Problem

## WARNING! This decision problem uses a modified version of run, and will have to be patched if the superclass is changed

class Sleeping_Beauty_V1(Decision_Problem):

    # Reward given by bet, should result in SIA/thirder

    def __init__(self):
        self.actions = ["GUESS HEADS", "GUESS TAILS"]

        self.states = ["Heads+Monday", "T+Monday", "Tails+Tuesday"]

        self.end_states = ["Heads+Monday", "Tails+Tuesday"]

        self.epistemic_states = ["Awake"]

        self.state_to_epistemic_state_dict = {"Heads+Monday":"Awake",
                                              "T+Monday":"Awake",
                                              "Tails+Tuesday":"Awake",
                                              "END":"END"
                                             }

        self.causation_dict = {("T+Monday", "GUESS HEADS"):"Tails+Tuesday",
                               ("T+Monday", "GUESS TAILS"):"Tails+Tuesday",
                              }

        self.utility_dict = {} # This decision problem uses a modified run function instead

        self.description = "Sleeping Beauty, reward by bet"

    def reset_with(self, agent):
        self.finished = False
        HEADS = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        if HEADS:
            self.state = "Heads+Monday"
        else:
            self.state = "T+Monday"

    def runLearn(self, agent, iterations, learn):
        history = []

        for i in range(iterations):

            self.reset_with(agent)
            episode = []

            while not self.finished:

                epistemic_state = self.epistemic_state()
                action_distribution = agent.get_action_distribution(epistemic_state)
                action = np.random.choice(self.actions, 1, p=action_distribution)[0]

                if self.state == "Heads+Monday":
                    utility = action_distribution[0]
                else:
                    utility = action_distribution[1]

                _,_ = self.do(action)

                episode.append((epistemic_state, action, utility))
                agent.learn_from([episode])

            history.append(episode)

        return history


class Sleeping_Beauty_V2(Decision_Problem):

    # Reward given by game, should result in SSA/halfer

    def __init__(self):
        self.actions = ["GUESS HEADS", "GUESS TAILS"]

        self.states = ["Heads+Monday", "T+Monday", "Tails+Tuesday"]

        self.end_states = ["Heads+Monday", "Tails+Tuesday"]

        self.epistemic_states = ["Awake"]

        self.state_to_epistemic_state_dict = {"Heads+Monday":"Awake",
                                              "T+Monday":"Awake",
                                              "Tails+Tuesday":"Awake",
                                              "END":"END"
                                             }

        self.causation_dict = {("T+Monday", "GUESS HEADS"):"Tails+Tuesday",
                               ("T+Monday", "GUESS TAILS"):"Tails+Tuesday",
                              }

        self.utility_dict = {} # This decision problem uses a modified run function instead

        self.description = "Sleeping Beauty, reward by game"

    def reset_with(self, agent):
        self.finished = False
        HEADS = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        if HEADS:
            self.state = "Heads+Monday"
        else:
            self.state = "T+Monday"

    def runLearn(self, agent, iterations, learn):
        history = []

        for i in range(iterations):

            self.reset_with(agent)
            episode = []

            while not self.finished:

                epistemic_state = self.epistemic_state()
                action_distribution = agent.get_action_distribution(epistemic_state)
                action = np.random.choice(self.actions, 1, p=action_distribution)[0]

                if self.state == "Heads+Monday":
                    utility = action_distribution[0]
                else:
                    utility = action_distribution[1]/2.0

                _,_ = self.do(action)

                episode.append((epistemic_state, action, utility))
                agent.learn_from([episode])

            history.append(episode)

        return history
