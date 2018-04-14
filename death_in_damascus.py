import numpy as np
from decision_problem import Decision_Problem
from agents import Simple_Agent, Softmax, Average

class Death_In_Damascus(Decision_Problem):
    def __init__(self):
        self.actions = ["STAY IN DAMASCUS", "FLEE TO ALEPPO"]

        self.states = ["DDamascus", "DAleppo", "ADamascus+DDamascus", "ADamascus+DAleppo", "AAleppo+DDamascus", "AAleppo+DAleppo"]

        self.end_states = ["ADamascus+DDamascus", "ADamascus+DAleppo", "AAleppo+DDamascus", "AAleppo+DAleppo"]

        self.epistemic_states = ["Death states he will come for you tomorrow", "END"]

        self.state_to_epistemic_state_dict = {"DDamascus":"Death states he will come for you tomorrow",
                                              "DAleppo":"Death states he will come for you tomorrow",
                                              "ADamascus+DDamascus":"END",
                                              "ADamascus+DAleppo":"END",
                                              "AAleppo+DDamascus":"END",
                                              "AAleppo+DAleppo":"END"
                                             }

        self.causation_dict = {("DDamascus", "STAY IN DAMASCUS"):"ADamascus+DDamascus",
                               ("DDamascus", "FLEE TO ALEPPO"):"AAleppo+DDamascus",
                               ("DAleppo", "STAY IN DAMASCUS"):"ADamascus+DAleppo",
                               ("DAleppo", "FLEE TO ALEPPO"):"AAleppo+DAleppo"
                              }

        self.utility_dict = {"ADamascus+DDamascus":0,
                             "ADamascus+DAleppo":5,
                             "AAleppo+DDamascus":5,
                             "AAleppo+DAleppo":0
                            }

        self.description = "Death in Damascus"

    def reset_with(self, agent):
        self.finished = False

        action_distribution = agent.get_action_distribution("Death states he will come for you tomorrow")
        action = np.random.choice(self.actions, 1, p=action_distribution)[0]

        if action == "STAY IN DAMASCUS":
             self.state = "DDamascus"
        else:
             self.state = "DAleppo"



decision_problem = Death_In_Damascus()
exploration_scheme = Softmax(0.5)
learning_scheme = Average()
agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem)
history=decision_problem.runLearn(agent,100, True)
print(history)
print(agent.get_action_distribution("Death states he will come for you tomorrow"))
