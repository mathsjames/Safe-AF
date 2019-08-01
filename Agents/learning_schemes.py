# These are used for decision problems with more than one step
# For 1-step decision problems the choice of function does not matter

class Identity_Function:
    def __init__(self):
        pass
    def process(self, history):
        return history

class Total:
    # Q-learning
    def __init__(self):
        pass
    def process(self, history):
        ans = []
        for episode in history:
            total = 0
            e = episode.copy()
            for step in e.steps[::-1]:
                total += step.reward
                step.reward = total
            ans.append(e)
        return ans
