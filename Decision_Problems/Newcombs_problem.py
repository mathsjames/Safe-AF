import numpy as np
from Decision_Problems.decision_problem import Decision_Problem

class Newcombs_Problem_V1(Decision_Problem):
    def __init__(self):
        self.actions = ["Take one box", "Take both boxes"]

        self.states = ["Opaque box full", "Opaque box empty",
                       "£15", "£10", "£5", "£0"]

        self.end_states = ["£15", "£10", "£5", "£0"]

        self.epistemic_states = ["making the choice", "END"]

        self.state_to_epistemic_state_dict = {"Opaque box full":"making the choice",
                                              "Opaque box empty":"making the choice",
                                              "£15":"END",
                                              "£10":"END",
                                              "£5":"END",
                                              "£0"::"END"
                                             }

        self.causation_dict = {("Opaque box full", "Take one box"):"",
                               ("Opaque box full", "Take both boxes"):"",
                               ("Opaque box empty", "Take one box"):"",
                               ("Opaque box empty", "Take both boxes"):""
                              }

        self.utility_dict = {"£15":15,
                             "£10":10,
                             "£5":5,
                             "£0":0
                            }

        self.description = "Newcomb's Problem"

    def reset_with(self, agent):
        self.finished = False

        action_distribution = agent.get_action_distribution("making the choice")
        action = np.random.choice(self.actions, 1, p=action_distribution)[0]

        if action == "Take one box":
             self.state = "Opaque box full"
        else:
             self.state = "Opaque box empty"



class Newcombs_Problem_V2(Decision_Problem):
    def __init__(self):
        self.actions = ["Take one box", "Take both boxes"]

        self.states = ["making the choice", "Have one box", "Have both boxes"]

        self.end_states = ["Have one box", "Have both boxes"]

        self.epistemic_states = ["making the choice", "END"]

        self.state_to_epistemic_state_dict = {"making the choice":"making the choice",
                                              "Have one box":"END",
                                              "Have both boxes":"END",
                                             }

        self.causation_dict = {("making the choice", "Take one box"):"Have one box",
                               ("making the choice", "Take both boxes"):"Have both boxes"
                              }

        self.description = "Newcomb's Problem"

    def reset_with(self, agent):
        self.finished = False

        action_distribution = agent.get_action_distribution("making the choice")

        10*action_distribution[0]

        self.utility_dict = {"Have one box":10*action_distribution[0],
                             "Have both boxes":1+10*action_distribution[0]}
