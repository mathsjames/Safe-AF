import numpy as np

from absent_minded_driver import Absent_Minded_Driver
from evidential_blackmail import Evidential_Blackmail
from agents import *

iterations = 5
epochs = 100000
batch_size = 1

AMD = Absent_Minded_Driver()
EB = Evidential_Blackmail()

softmax = Softmax(0.1)
epsilongreedy = Epsilon_Greedy(0.01)

average = Average()
idf = Identity_Function()

test_configs = [("Softmax + Average", AMD, softmax, average, ["Intersection"]),
                ("Epsilon Greedy + Average", AMD, epsilongreedy, average, ["Intersection"]),
                ("Softmax + ID", EB, softmax, idf, ["Blackmail", "No Blackmail"]),
                ("Epsilon Greedy + ID", EB, epsilongreedy, idf, ["Blackmail", "No Blackmail"]),
                ("Softmax + ID", DiD, epsilongreedy, idf, ["Blackmail", "No Blackmail"]),
                ("Epsilon Greedy + ID", DiD, epsilongreedy, idf, ["Death states he will come for you tomorrow"])]

for agent_description, decision_problem, exploration_scheme, learning_scheme, interesting_states in test_configs:

    print(decision_problem.description)
    print(agent_description)

    for i in range(iterations):

        agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem)

        for j in range(epochs):
            history = decision_problem.run(agent, batch_size)
            agent.learn_from(history)

        print("Average utility: " + str(agent.total_utility/agent.games_played))

        for state in interesting_states:
            print(state + ":")
            #print(agent.get_action_distribution(state))

        print()

    print("###")
