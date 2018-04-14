import numpy as np

class Decision_Problem:
    
    def __init__(self):   
        self.actions = []
        self.states = []
        self.end_states = []
        self.epistemic_states = []
        
        self.state_to_epistemic_state_dict = {}
    
        self.causation_dict = {}

        self.utility_dict = {}

    def reset_with(self, agent):
        self.finished = False
        self.state = None
    
    #runs and lets the agent learn after each instance of the problem if input variable learn is True.
    def runLearn(self, agent, iterations, learn):
        history = []
        
        for i in range(iterations):          

            self.reset_with(agent)
            episode = []
            
            while not self.finished:
                
                epistemic_state = self.epistemic_state()
                action_distribution = agent.get_action_distribution(epistemic_state)
                print(action_distribution)
                action = np.random.choice(self.actions, 1, p=action_distribution)[0]
                _, utility = self.do(action)
                
                episode.append((epistemic_state, action, utility))
                agent.learn_from([episode])
                
            history.append(episode)
            
        return history
    
    def run(self, agent, iterations):
        return run(self, agent, iterations, False)
        
            
    def do(self, action):
        
        self.state = self.cause(self.state, action)
        utility = self.utility(self.state)
            
        if self.state in self.end_states:
            self.finished = True
        
        return (self.epistemic_state(), utility)
        
    def utility(self, state):
        try:
            return self.utility_dict[state]
        except:
            return 0
    
    def epistemic_state(self):
        return self.state_to_epistemic_state_dict[self.state]
        
    def cause(self, state, action):
        try:
            return self.causation_dict[(state, action)]
        except:
            return state     
