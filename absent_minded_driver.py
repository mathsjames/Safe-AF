import numpy as np
from decision_problem import Decision_Problem

class Absent_Minded_Driver(Decision_Problem):
    
    def __init__(self):   
        self.actions = ["CONT", "EXIT"]
        self.states = ["Intersection_1", "Intersection_2", "Stop_A", "Stop_B", "Stop_C"]
        self.end_states = ["Stop_A", "Stop_B", "Stop_C"]
        self.epistemic_states = ["Intersection", "Stop_A", "Stop_B", "Stop_C"]
        
        self.state_to_epistemic_state_dict = {"Intersection_1":"Intersection",
                                              "Intersection_2":"Intersection", 
                                              "Stop_A":"Stop_A",
                                              "Stop_B":"Stop_B",
                                              "Stop_C":"Stop_C",
                                             }
    
        self.causation_dict = {("Intersection_1", "EXIT"):"Stop_A", 
                               ("Intersection_1", "CONT"):"Intersection_2", 
                               ("Intersection_2", "EXIT"):"Stop_B",
                               ("Intersection_2", "CONT"):"Stop_C",
                              }

        self.utility_dict = {"Stop_A":0,
                             "Stop_B":4,
                             "Stop_C":1,
                            }

    def reset_with(self, agent):
        self.finished = False
        self.state = "Intersection_1"   
  
