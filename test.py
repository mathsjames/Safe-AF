import numpy as np

from absent_minded_driver import Absent_Minded_Driver
from evidential_blackmail import Evidential_Blackmail
from general2by2game import General2by2
from death_in_damascus import Death_In_Damascus
from prisoners_dilemma_against_copy import Prisoners_Dilemma_against_copy
from sleeping_beauty import *
from conitzer import Conitzer
from agents import *
import matplotlib.pyplot as plt

repetitions = 10 # for testing stability
iterations = 100000
#epochs = 1000
#batch_size = 1

AMD = Absent_Minded_Driver()
EB = Evidential_Blackmail()
G2EB = General2by2(lambda dummy: [(10,0),(11,1)],lambda dist: dist, "Evidential Blackmail" )
DiD = Death_In_Damascus()
G2DiD = General2by2(lambda dummy: [(0,5),(5,0)],lambda dist: dist, "Death in Damascus" )
G2NPR = General2by2(lambda dist: [(dist[0]*10,dist[0]*10),(dist[0]*10+1,dist[0]*10+1)],lambda dist: [1,0], "Newcombs problem with rewared proportional to 1 box probability" )
PDS = Prisoners_Dilemma_against_copy()
SB1 = Sleeping_Beauty_V1()
SB2 = Sleeping_Beauty_V2()
Conitzer = Conitzer()

softmax = Softmax(1)
epsilongreedy = Epsilon_Greedy(0.01)
xpcooling = exponential_cooling()
Asoftmax = More_Advanced_Softmax(xpcooling)

total = Total()
average = Average()
idf = Identity_Function()

test_configs = [#("Softmax + Total", AMD, softmax, total, ["Intersection"]),
                #("Epsilon Greedy + Average", AMD, epsilongreedy, average, ["Intersection"]),
                #("Softmax + ID", AMD, softmax, idf, ["Intersection"]),
                #("Epsilon Greedy + ID", AMD, epsilongreedy, idf, ["Intersection"]),
                #("Softmax", EB, softmax, idf, ["Blackmail", "No Blackmail"]),
                #("Epsilon Greedy", EB, epsilongreedy, idf, ["Blackmail", "No Blackmail"]),
                #("Softmax", SB1, softmax, idf, ["Awake"]),
                #("Softmax", SB2, softmax, idf, ["Awake"]),
                #("Softmax", DiD, softmax, idf, ["Death states he will come for you tomorrow"]),
                #("Epsilon Greedy", DiD, epsilongreedy, idf, ["Death states he will come for you tomorrow"]),
                #("Epsilon Greedy", G2DiD, epsilongreedy, idf, ["NewRound"]),
                ("Softmax", PDS, softmax, idf, ["START"]),
                #("Epsilon Greedy", PDS, epsilongreedy, average, ["START"]),
                #("Softmax", G2EB, softmax, average, ["NewRound"]),
                #("Softmax", G2NPR, softmax, average, ["NewRound"])
                ("Softmax", Conitzer, softmax, average, ["White", "Black", "Grey", "START"]),
                ("Epsilon Greedy", Conitzer, epsilongreedy, average, ["White", "Black", "Grey", "START"])
]

for agent_description, decision_problem, exploration_scheme, learning_scheme, interesting_states in test_configs:

    print(decision_problem.description)
    print(agent_description)

    distribution_histories = []

    for i in range(repetitions):

        agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem)

        history, distribution_history, EXP_history = decision_problem.run(agent, iterations, learn=True, interesting_states=interesting_states)
        distribution_histories.append(distribution_history)
        #for j in range(epochs):
            #history = decision_problem.run(agent, batch_size, learn=True)
            #agent.learn_from(history)

        print("Average utility: " + str(agent.total_utility/agent.games_played))
        #print(agent.games_played)

        for state in interesting_states:
            print(state + ":")
            for i in range(len(agent.actions)):
                a = agent.actions[i]
                p = agent.get_action_distribution(state)[i]
                print(str(a) + ": " + str(p))

                #for state in interesting_states:

        print()

    for state in interesting_states:
        for distribution_history in distribution_histories:
            plt.plot(distribution_history[state])

        plt.title("Probability of " + str(agent.actions[0]) + " when " + state + " in " + decision_problem.description)
        plt.show(decision_problem.description + state + " action distributions")
        #plt.savefig("C:\Users\Joar\Desktop\Decision Problems\plot "+decision_problem.description + state + " action distributions.png")

        for distribution_history in distribution_histories:
            plt.plot(EXP_history[state][0])
        plt.title("EXP of " + agent.actions[0] + " when " + state + " in " + decision_problem.description)
        plt.show(decision_problem.description + state + " EXP of " + agent.actions[0])
        #plt.savefig("C:\Users\Joar\Desktop\Decision Problems\plot "+decision_problem.description + state + " EXP of " + agent.actions[0] + ".png")

        for distribution_history in distribution_histories:
            plt.plot(EXP_history[state][1])
        plt.title("EXP of " + agent.actions[1] + " when " + state + " in " + decision_problem.description)
        plt.show(decision_problem.description + state + " EXP of " + agent.actions[1])
        #plt.savefig("C:\Users\Joar\Desktop\Decision Problems\plot "+ decision_problem.description + state + " EXP of " + agent.actions[1] + ".png")

    print("###")
