import numpy as np
from decision_problem import Decision_Problem

class Evidential_Blackmail(Decision_Problem):

    # Full specification of the absent-minded driver problem

    def __init__(self):
        self.actions = ["PAY", "DONT"]

        self.states = ["crash+blackmail", "no crash+blackmail",
                       "crash+no blackmail", "no crash+no blackmail",
                       "crash+pay", "crash+no pay", "no crash+pay", "no crash+no pay"]

        self.end_states = ["crash+pay", "crash+no pay", "no crash+pay", "no crash+no pay"]

        self.epistemic_states = ["Blackmail", "No Blackmail", "END"]

        self.state_to_epistemic_state_dict = {"crash+blackmail":"Blackmail",
                                              "no crash+blackmail":"Blackmail",
                                              "crash+no blackmail":"No Blackmail",
                                              "no crash+no blackmail":"No Blackmail",
                                              "crash+pay":"END",
                                              "crash+no pay":"END",
                                              "no crash+pay":"END",
                                              "no crash+no pay":"END",
                                             }

        self.causation_dict = {("crash+blackmail", "PAY"):"crash+pay",
                               ("crash+blackmail", "DONT"):"crash+no pay",
                               ("no crash+blackmail", "PAY"):"no crash+pay",
                               ("no crash+blackmail", "DONT"):"no crash+no pay",
                               ("crash+no blackmail", "PAY"):"crash+pay",
                               ("crash+no blackmail", "DONT"):"crash+no pay",
                               ("no crash+no blackmail", "PAY"):"no crash+pay",
                               ("no crash+no blackmail", "DONT"):"no crash+no pay",
                              }

        self.utility_dict = {"no crash+no pay":15,
                             "no crash+pay":10,
                             "crash+no pay":5,
                             "crash+pay":0
                            }
        self.description = "Evidential_Blackmail"

    def reset_with(self, agent):
        self.finished = False
        crash = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]

        if crash:
            action_distribution = agent.get_action_distribution("Blackmail")
            action = np.random.choice(self.actions, 1, p=action_distribution)[0]

            if action == "PAY":
                self.state = "crash+no blackmail"
            else:
                self.state = "crash+blackmail"

        else:
            action_distribution = agent.get_action_distribution("Blackmail")
            action = np.random.choice(self.actions, 1, p=action_distribution)[0]

            if action == "PAY":
                self.state = "no crash+blackmail"
            else:
                self.state = "no crash+no blackmail"
