import numpy as np
import matplotlib.pyplot as plt

from customizable_prisoners_dilemma import customisable_prisoners_dilemma
from agents import *

iterations = 100000
repetitions = 10

softmax = Softmax(1)
epsilongreedy = Epsilon_Greedy(0.01)
xpcooling = exponential_cooling()
Asoftmax = More_Advanced_Softmax(xpcooling)

idf = Identity_Function()

exploration_scheme = softmax
learning_scheme = idf

x = []
y = []

for ratio in np.arange(0.0, 1.0, 0.01):

    print(ratio)
    ys = []

    for i in range(repetitions):

        prisoners_dilemma = customisable_prisoners_dilemma(ratio)
        agent = Simple_Agent(exploration_scheme, learning_scheme, prisoners_dilemma)
        _, distribution_history, _ = prisoners_dilemma.run(agent, iterations, learn=True, interesting_states=["START"])
        ys.append(np.mean(distribution_history["START"][int(0.8*iterations):]))
    #print(distribution_history)
    x.append(ratio)
    y.append(np.mean(ys))

plt.plot(x, y)
plt.title("Probability of cooperate by ratio of reward")
plt.show()
