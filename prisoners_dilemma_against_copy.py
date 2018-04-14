import numpy as np
from decision_problem import Decision_Problem

class Prisoners_Dilemma_against_copy(Decision_Problem):

    # Agent is playin Prisoners Dilema against a frech copy of it self.

    # The choices are to defect (take 1 utility for it self) or cooperate (give 10 utility to its coppy)

    def __init__(self):

        self.actions = ["Cooperate", "Defect"]

        self.states = ["copy cooperates", "copy defects",
                    "cooperate+copy cooperates", "cooperate+copy defects", "defect+copy coperates", "defect+copy defects"]

        self.end_states = ["cooperate+copy cooperates", "cooperate+copy defects", "defect+copy coperates", "defect+copy defects"]

        self.epistemic_states = ["START", "END"]

        self.state_to_epistemic_state_dict = {"copy cooperates":"START",
                                              "copy defects":"START",
                                              "cooperate+copy cooperates":"END",
                                              "cooperate+copy defects":"END",
                                              "defect+copy coperates":"END",
                                              "defect+copy defects":"END"}

        self.causation_dict = {("copy cooperates", "Cooperate"):"cooperate+copy cooperates",
                               ("copy cooperates", "Defect"):"defect+copy coperates",
                               ("copy defects", "Cooperate"):"cooperate+copy defects",
                               ("copy defects", "Defect"):"defect+copy defects"}

        self.utility_dict = {"cooperate+copy cooperates":10,
                             "cooperate+copy defects":0,
                             "defect+copy coperates":11,
                             "defect+copy defects":1}

        self.description = "Prisoner's Dilemma Against Copy"


    def reset_with(self, agent):
        self.finished = False

        action_distribution = agent.get_action_distribution("START")
        action = np.random.choice(self.actions, 1, p=action_distribution)[0]

        if action == "Cooperate":
            self.state = "copy cooperates"

        if action == "Defect":
            self.state = "copy defects"
