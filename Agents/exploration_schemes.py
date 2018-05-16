import numpy as np

class Epsilon_Greedy:

    def __init__(self, epsilon):
        self.epsilon = float(epsilon)

    def function(self, x, _):
        exploring_distribution = [self.epsilon/len(x) for i in x]

        mx = max(x)
        mxs = list(filter(lambda x: x == mx, x))
        not_exploring_distribution=[(1-self.epsilon)/len(mxs) if i==mx else 0 for i in x]

        ans = [a+b for (a,b) in zip(exploring_distribution, not_exploring_distribution)]

        return ans

class Epsilon_First:

    def __init__(self, epsilon):
        self.epsilon = float(epsilon)

    def function(self, x, games_played):
        if games_played < 1000:
            return [1/len(x) for i in x]
        else:
            mx = max(x)
            mxs = list(filter(lambda x: x == mx, x))
            distribution=[1/len(mxs) if i==mx else 0 for i in x]
            return distribution

class Epsilon_Decreasing:

    def __init__(self, cooling_function):
        self.cooling_function = cooling_function

    def function(self, x, games_played):

        epsilon = self.cooling_function.function(games_played)

        exploring_distribution = [epsilon/len(x) for i in x]

        mx = max(x)
        mxs = list(filter(lambda x: x == mx, x))
        not_exploring_distribution=[(1-epsilon)/len(mxs) if i==mx else 0 for i in x]

        ans = [a+b for (a,b) in zip(exploring_distribution, not_exploring_distribution)]

        return ans

class Softmax:

    def __init__(self, temperature):
        self.temperature = float(temperature)

    def function(self, x, _):
        x = [i/self.temperature for i in x]
        e_x = np.exp(x-np.max(x))
        return e_x / e_x.sum(axis=0)

class Cooling_Softmax:

    def __init__(self, cooling_function):
        self.cooling_function = cooling_function

    def function(self, x, games_played):
        temperature = self.cooling_function.function(games_played)
        if temperature==0:
            mx = max(x)
            mxs = list(filter(lambda x: x == mx, x))
            return [1.0/len(mxs) if i==mx else 0 for i in x]
        else:
            x = [i/temperature for i in x]
            e_x = np.exp(x-np.max(x))
            return e_x / e_x.sum(axis=0)

## Cooling functions ##

# A cooling function should be a function f: # games played -> temperature

class Exponential_cooling:
    def __init__(self, initial_value):
        self.v = initial_value
    def function(self, games_played):
        return self.v*(0.99**games_played)

class Lambda_cooling:
    def __init__(self,func):
        self.func = func
    def function(self, games_played):
        return self.func(games_played)
