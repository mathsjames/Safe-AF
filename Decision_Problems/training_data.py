class Episode:
    def __init__(self, utility, steps):
        self.utility = utility # number
        self.steps = steps # should be a list of steps

    def copy(self):
        ans_steps = [step.copy() for step in self.steps]
        ans = Episode(self.utility, ans_steps)
        return ans

class Step:
    def __init__(self, epistemic_state, action, reward):
        self.epistemic_state = epistemic_state
        self.action = action
        self.reward = reward

    def copy(self):
        ans = Step(self.epistemic_state, self.action, self.reward)
        return ans
