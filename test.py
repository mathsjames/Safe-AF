import numpy as np

from absent_minded_driver import Absent_Minded_Driver
from evidential_blackmail import Evidential_Blackmail
from general2by2game import General2by2
from death_in_damascus import Death_In_Damascus
from prisoners_dilemma_against_copy import Prisoners_Dilemma_against_copy
from sleeping_beauty import *
from agents import *

iterations = 1
epochs = 1000
batch_size = 1

AMD = Absent_Minded_Driver()
EB = Evidential_Blackmail()
G2EB = General2by2([(10,0),(11,1)],lambda dist: dist )
DiD = Death_In_Damascus()
PDS = Prisoners_Dilemma_against_copy()
SB1 = Sleeping_Beauty_V1()
SB2 = Sleeping_Beauty_V2()

softmax = Softmax(0.1)
epsilongreedy = Epsilon_Greedy(0.01)

average = Average()
idf = Identity_Function()

test_configs = [("Softmax + Average", AMD, softmax, average, ["Intersection"]),
                ("Epsilon Greedy + Average", AMD, epsilongreedy, average, ["Intersection"]),
                ("Softmax + ID", AMD, softmax, idf, ["Intersection"]),
                ("Epsilon Greedy + ID", AMD, epsilongreedy, idf, ["Intersection"]),
                ("Softmax", EB, softmax, idf, ["Blackmail", "No Blackmail"]),
                ("Epsilon Greedy", EB, epsilongreedy, idf, ["Blackmail", "No Blackmail"]),
                ("Softmax", SB1, softmax, average, ["Awake"]),
                ("Epsilon Greedy", SB1, epsilongreedy, average, ["Awake"]),
                ("Softmax", SB2, softmax, average, ["Awake"]),
                ("Epsilon Greedy", SB2, epsilongreedy, average, ["Awake"]),
                #("Softmax", DiD, epsilongreedy, idf, ["Death states he will come for you tomorrow"]),
                #("Epsilon Greedy", DiD, epsilongreedy, idf, ["Death states he will come for you tomorrow"]),
                #("Softmax", PDS, softmax, average, ["START"]),
                #("Epsilon Greedy", PDS, epsilongreedy, average, ["START"]),
                ("Softmax", G2EB, softmax, average, ["NewRound"])
]

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
            for i in range(len(agent.actions)):
                a = agent.actions[i]
                p = agent.get_action_distribution(state)[i]
                print(a + ": " + str(p))

        print()

    print("###")
