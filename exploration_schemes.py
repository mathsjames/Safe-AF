import numpy as np

class Epsilon_Greedy:

    def __init__(self, epsilon):
        self.epsilon = float(epsilon)

    def function(self, x):
        exploring_distribution = [self.epsilon/len(x) for i in x]
        
        mx = max(x)
        mxs = list(filter(lambda x: x == mx, x))
        not_exploring_distribution=[(1-self.epsilon)/len(mxs) if i==mx else 0 for i in x]

        ans = [a+b for (a,b) in zip(exploring_distribution, not_exploring_distribution)]

        return ans

# Epsilon-First?
# Epsilon Decreasing?

class Softmax:

    def __init__(self, temperature):
        self.temperature = float(temperature)

    def function(self, x):
        x = [i/self.temperature for i in x]
        e_x = np.exp(x-np.max(x))
        return e_x / e_x.sum(axis=0)

class More_Advanced_Softmax:

    def __init__(self, cooling_function):
        self.cooling_function = cooling_function
        #Cooling function should be a function from number of games played to temperature

    def function(self, x, games_played):
        temperature = self.cooling_function.function(games_played)
        if temperature==0:
            return [1.0/len(mxs) if i==mx else 0 for i in x]
        else:
            x = [i/temperature for i in x]
            e_x = np.exp(x-np.max(x))
            return e_x / e_x.sum(axis=0)

#def exponential_cooling(games_played):
#    # Example of cooling function
#    if  100*(0.99**games_played) > 0.00001:
#        temperature = 100*(0.99**games_played)
#    else:
#        temperature = 0.00001
#    return temperature

class Lambda_cooling:
    def __init__(self,func):
        self.func=func

    def function(self, games_played):
        return self.func(games_played)

class Exponential_cooling:
    def function(self, games_played):
        temperature = 1000*(0.99**games_played)
        return temperature
