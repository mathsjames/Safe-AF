import numpy as np
from decision_problem import Decision_Problem

class Death_In_Damascus(Decision_Problem):
    def __init__(self):
        self.actions = ["STAY", "FLEE"]

        self.states = ["DDamascus", "DAleppo", "ADamascus+DDamascus", "ADamascus+DAleppo", "AAleppo+DDamascus", "AAleppo+DAleppo"]

        self.end_states = ["ADamascus+DDamascus", "ADamascus+DAleppo", "AAleppo+DDamascus", "AAleppo+DAleppo"]

        self.epistemic_states = ["warned by death", "END"]

        self.state_to_epistemic_state_dict = {"DDamascus":"warned by death",
                                              "DAleppo":"warned by death",
                                              "ADamascus+DDamascus":"END",
                                              "ADamascus+DAleppo":"END",
                                              "AAleppo+DDamascus":"END",
                                              "AAleppo+DAleppo":"END"
                                             }

        self.causation_dict = {("DDamascus", "STAY"):"ADamascus+DDamascus",
                               ("DDamascus", "FLEE"):"AAleppo+DDamascus",
                               ("DAleppo", "STAY"):"ADamascus+DAleppo",
                               ("DAleppo", "FLEE"):"AAleppo+DAleppo"
                              }

        self.utility_dict = {"ADamascus+DDamascus":0,
                             "ADamascus+DAleppo":5,
                             "AAleppo+DDamascus":5,
                             "AAleppo+DAleppo":0
                            }

        self.description = "Death in Damascus"

    def reset_with(self, agent):
        self.finished = False

        action_distribution = agent.get_action_distribution("warned by death")
        action = np.random.choice(self.actions, 1, p=action_distribution)[0]

        if action == "STAY":
             self.state = "DDamascus"
        else:
             self.state = "DAleppo"


'''
decision_problem = Death_In_Damascus()
exploration_scheme = Softmax(0.5)
learning_scheme = Average()
agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem)
history=decision_problem.run(agent,100000, True)
print(history)
print(agent.get_action_distribution("Death states he will come for you tomorrow"))
'''
