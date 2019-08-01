import numpy as np
from Decision_Problems.decision_problem import Decision_Problem
from Decision_Problems.training_data import *
import math
#import matplotlib.pyplot as plt

## WARNING! This decision problem has a  non-standard internal structure

class Sleeping_Beauty_by_bet(Decision_Problem):

    # Reward given by bet, should result in SIA/thirder

    def __init__(self):
        self.actions = ["GUESS HEADS", "GUESS TAILS"]

        self.states = [(coin, day, bet) for coin in ["Heads","Tails"]
                                        for day in ["Monday", "Tuesday", "END"]
                                        for bet in ["Heads","Tails", "_"]
                                        ]

        self.epistemic_states = ["Awake","END"]

        self.description = "Sleeping Beauty, reward by bet"

    def reset_with(self, agent):
        self.finished = False
        HEADS = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
        if HEADS:
            self.state = ("Heads", "Monday", "_")
        else:
            self.state = ("Tails", "Monday", "_")

    def utility(self, state):
        coin, day, bet = state
        if coin == bet:
            return 1
        else:
            return 0

    def epistemic_state(self):
        coin, day, bet = self.state
        if day == "END":
            return "END"
        else:
            return "Awake"

    def cause(self, state, action):

        coin, day, bet = state

        if action == "GUESS HEADS":
            bet = "Heads"
        elif action == "GUESS TAILS":
            bet = "Tails"
        else:
            print("Something is wrong in sleeping beauty cause")

        if (day == "Monday" and coin == "Heads") or (day == "Tuesday"):
            day = "END"
        elif (day == "Monday" and coin == "Tails"):
            day = "Tuesday"
        else:
            print("Something is wrong in sleeping beauty cause")

        return coin, day, bet

    def is_final_state(self, state):
        coin, day, bet = state
        return day == "END"


class Sleeping_Beauty_by_game(Sleeping_Beauty_by_bet):

    # Reward given by game, should result in SSA/halfer

    def __init__(self):
        Sleeping_Beauty_by_bet.__init__(self)
        self.description = "Sleeping Beauty, reward by game"

    def play(self, agent):

            self.reset_with(agent)
            while not self.finished:

                epistemic_state = self.epistemic_state()
                action_distribution = agent.get_action_distribution(epistemic_state)
                action = np.random.choice(self.actions, 1, p=action_distribution)[0]
                _, reward = self.do(action)
                step = Step(epistemic_state, action, reward)

            episode = Episode(reward, [step])
            return episode
