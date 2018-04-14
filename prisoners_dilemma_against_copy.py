import numpy as np
from decision_problem import Decision_Problem

class Prisoners_Dilema_agians_coppy(Decision_Problem):

    # Agent is playin Prisoners Dilema against a frech copy of it self. 
    
    # The choices are to defect (take 1 utility for it self) or cooperate (give 10 utility to its coppy)
    
    def __init__(self):
        
        self.actions["Cooperate", "Defect"]
        
        self.states["coppy cooperates", "coppy defects",
                    "cooperate+coppy cooperates", "cooperate+coppy defects", "defect+coppy coperates", "defect+coppy defects"] 
        
        self.end_states = ["cooperate+coppy cooperates", "cooperate+coppy defects", "defect+coppy coperates", "defect+coppy defects"]
            
        self.epistemic_states = ["START", "END"]
        
        self.state_to_epistemic_state_dict = {"coppy cooperates":"START", 
                                              "coppy defects":"START",
                                              "cooperate+coppy cooperates":"END", 
                                              "cooperate+coppy defects":"END",
                                              "defect+coppy coperates":"END",
                                              "defect+coppy defects":"END"}
                                              
        self.causation_dict = {("coppy cooperates", "Cooperate"):"cooperate+coppy cooperates", 
                               ("coppy cooperates", "Defect"):"defect+coppy coperates", 
                               ("coppy defects", "Cooperate"):"cooperate+coppy defects", 
                               ("coppy defects", "Defect"):"defect+coppy defects"}
                              
        self.utility_dict = {"cooperate+coppy cooperates":10, 
                             "cooperate+coppy defects":0,
                             "defect+coppy coperates":11,
                             "defect+coppy defects":1}
        
                             
        def reset_with(self, agent):
            self.finished = False
            
            action_distribution = agent.get_action_distribution("Start")
            action = np.random.choice(self.actions, 1, p=action_distribution)[0]
            
            if action == "Cooperate":
                self.state = "coppy cooperates"
                
            if action == "Defect":
                self.state = "coppy defects"
                                              
                                              
        
        
        
        
