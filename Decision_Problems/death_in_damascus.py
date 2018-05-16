import numpy as np
from Decision_Problems.decision_problem import Decision_Problem

class Death_In_Damascus(Decision_Problem):
    def __init__(self):
        self.actions = ["STAY", "FLEE"]

        self.states = ["Death in Damascus", "Death in Aleppo", "Same location", "Different locations"]

        self.end_states = ["Same location", "Different locations"]

        self.epistemic_states = ["warned by death", "END"]

        self.state_to_epistemic_state_dict = {"Death in Damascus":"warned by death",
                                              "Death in Aleppo":"warned by death",
                                              "Same location":"END",
                                              "Different locations":"END",
                                             }

        self.causation_dict = {("Death in Damascus", "STAY"):"Same location",
                               ("Death in Damascus", "FLEE"):"Different locations",
                               ("Death in Aleppo", "STAY"):"Different locations",
                               ("Death in Aleppo", "FLEE"):"Same location"
                              }

        self.utility_dict = {"Same location":0,
                             "Different locations":5,
                            }

        self.description = "Death in Damascus"

    def reset_with(self, agent):
        self.finished = False

        action_distribution = agent.get_action_distribution("warned by death")
        action = np.random.choice(self.actions, 1, p=action_distribution)[0]

        if action == "STAY":
             self.state = "Death in Damascus"
        else:
             self.state = "Death in Aleppo"
