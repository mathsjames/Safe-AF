import numpy as np

from absent_minded_driver import Absent_Minded_Driver
from evidential_blackmail import Evidential_Blackmail
from agents import *

iterations = 5
epochs = 10000
batch_size = 1

AMD = Absent_Minded_Driver()
EB = Evidential_Blackmail()

softmax = Softmax(100)
epsilongreedy = Epsilon_Greedy(0.01)

average = Average()
idf = Identity_Function()

test_configs = [(AMD, softmax, average),
                (AMD, epsilongreedy, average),
                (EB, softmax, idf),
                (EB, epsilongreedy, idf)]
    
for decision_problem, exploration_scheme, learning_scheme in test_configs:

    for i in range(iterations):
        
        agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem)
        
        total_utility = 0
        
        for j in range(epochs):
            history = decision_problem.run(agent, batch_size)
            agent.learn_from(history)

        print(agent.total_utility)
        # optimal value: 4/3

        print(agent.get_action_distribution())
        # optimal value: 2/3
        
        print()
