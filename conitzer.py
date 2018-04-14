import numpy as np
import itertools
from decision_problem import Decision_Problem
from agents import Simple_Agent, Softmax, Average


#A description of the anthropic/DT problem described by Conitzer as a counterexample to DT+double-halfer here: https://users.cs.duke.edu/~conitzer/dutchSYNTHESE.pdf
class Conitzer(Decision_Problem):

    # Full specification of the absent-minded driver problem

    def __init__(self):
        self.actions = [True, False]

        color1 = ["Black", "White"]
        color2 = ["Black", "White", "Grey"]

        self.states = list(itertools.product(
            itertools.product(color1,color2),
            [()] +
            list(itertools.product(self.actions)) +
            list(itertools.product(self.actions, self.actions)) +
            list(itertools.product(self.actions, self.actions, self.actions))
            ))

        self.end_states = list(itertools.product(itertools.product(color1,color2),
            itertools.product(self.actions, self.actions, self.actions)))

        self.epistemic_states = ["START"]+ color2 + ["END"]

        self.state_to_epistemic_state_dict = {}
        for state in self.states:
            if len(state[1]) == 3:
                self.state_to_epistemic_state_dict[state]="END"
            elif len(state[1]) == 0:
                self.state_to_epistemic_state_dict[state]="START"
            else:
                print(state)
                self.state_to_epistemic_state_dict[state]=state[0][len(state[1])-1]

        self.causation_dict = {}
        for state in self.states:
            if len(state[1])<3:
                for action in self.actions:
                    self.causation_dict[(state,action)]=(state[0],state[1]+(action,))

        self.utility_dict = {}
        for state in self.end_states:
            utility = 0
            if state[1][0]:
                utility-=20 #the first bet costs 20...
                if state[0][1] == "Grey":
                    utility += 42 #... and pays out 42 if coin 2 comes up Grey
            if state[1][1]:
                utility -=24#the second bet costs 24...
                if state[0][1] != "Grey":
                    utility +=33#... and pays out 33 if coin 2 comes up Opposite (i.e., not Gray)
            if state[1][2] & state[1][2]!="Grey": #It is not offered in the grey room
                utility -=24
            self.utility_dict[state]=utility

        self.description = "ConitzerDevastatingExamples"

    def reset_with(self, agent):
        self.finished = False
        coin1 = np.random.choice(["Black", "White"], 1, p=[0.5, 0.5])[0]
        coin2 = np.random.choice(["Grey", "Opposite"], 1, p=[0.5, 0.5])[0]
        if coin2 == "Opposite":
            if coin1 == "Black":
                coin2 = "White"
            else:
                coin2 = "Black"

        self.state = ((coin1,coin2),())



decision_problem = Conitzer()
exploration_scheme = Softmax(0.1)
learning_scheme = Average()
agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem)
history = decision_problem.run(agent,1000000, True)
print(history)
print(agent.get_action_distribution("White"))
print(agent.get_action_distribution("Black"))
print(agent.get_action_distribution("Grey"))
print(agent.get_action_distribution("START"))
