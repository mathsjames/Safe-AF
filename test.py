import numpy as np
import matplotlib.pyplot as plt

from absent_minded_driver import Absent_Minded_Driver
from evidential_blackmail import Evidential_Blackmail
from general2by2game import General2by2
from death_in_damascus import Death_In_Damascus
from prisoners_dilemma_against_copy import Prisoners_Dilemma_against_copy
from sleeping_beauty import *
from conitzer import Conitzer
from agents import *
from exploration_schemes import *

repetitions = 10 # for testing stability
iterations = 1000
#epochs = 1000
#batch_size = 1

AMD = Absent_Minded_Driver()
EB = Evidential_Blackmail()
G2EB = General2by2(lambda dummy: [(10,0),(11,1)],lambda dist: dist, "Evidential Blackmail" )
DiD = Death_In_Damascus()
G2DiD = General2by2(lambda dummy: [(0,5),(5,0)],lambda dist: dist, "Death in Damascus" )
G2NPR = General2by2(lambda dist: [(dist[0]*10,dist[0]*10),(dist[0]*10+1,dist[0]*10+1)],lambda dist: [1,0], "Newcombs problem with rewared proportional to 1 box probability" )
PDS = Prisoners_Dilemma_against_copy()
SB_bet = Sleeping_Beauty_by_bet()
SB_game = Sleeping_Beauty_by_game()
Conitzer = Conitzer()
SH = General2by2(lambda dummy: [(2,0),(1,1)],lambda dist: dist, "Stag Hunt")

softmax = Softmax(0.1)
epsilongreedy = Epsilon_Greedy(0.01)
xpcooling = exponential_cooling()
cooling_softmax = More_Advanced_Softmax(xpcooling)
#better = Better()

total = Total()
average = Average()
idf = Identity_Function()

test_configs = [#("Softmax + Total", AMD, 5, softmax, total, ["Intersection"]),
                #("Epsilon Greedy + Average", AMD, 5, epsilongreedy, average, ["Intersection"]),
                #("Softmax + ID", AMD, softmax, 5, idf, ["Intersection"]),
                #("Epsilon Greedy + ID", AMD, epsilongreedy, 5, idf, ["Intersection"]),
                #("Softmax", EB, 5, softmax, idf, ["Blackmail", "No Blackmail"]),
                #("Epsilon Greedy", EB, 5, epsilongreedy, idf, ["Blackmail", "No Blackmail"]),
                #("Softmax", SB_bet, 1, softmax, idf, ["Awake"]),
                #("Softmax", SB_game, 1, softmax, idf, ["Awake"]),
                #("Softmax", DiD, 20, softmax, idf, ["Death states he will come for you tomorrow"]),
                #("Epsilon Greedy", 20, DiD, epsilongreedy, idf, ["Death states he will come for you tomorrow"]),
                #("Epsilon Greedy", 20, G2DiD, epsilongreedy, idf, ["NewRound"]),
                #("Softmax", PDS, 20, softmax, idf, ["START"]),
                #("Epsilon Greedy", PDS, 20, epsilongreedy, average, ["START"]),
                #("Softmax", G2EB, 20, softmax, average, ["NewRound"]),
                #("Softmax", G2NPR, 20, softmax, average, ["NewRound"])
                #("Softmax", Conitzer, 20, softmax, average, ["White", "Black", "Grey", "START"]),
                #("Epsilon Greedy", Conitzer, 20, epsilongreedy, average, ["White", "Black", "Grey", "START"])
]

for agent_description, decision_problem, prior, exploration_scheme, learning_scheme, interesting_states in test_configs:

    print(decision_problem.description)
    print(agent_description)

    distribution_histories = []

    for i in range(repetitions):

        agent = Simple_Agent(exploration_scheme, learning_scheme, decision_problem, prior)


        history, distribution_history, EXP_history = decision_problem.run(agent, iterations, learn=True, interesting_states=interesting_states)
        distribution_histories.append(distribution_history)
        #for j in range(epochs):
            #history, _, _ = decision_problem.run(agent, batch_size, learn=True)
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
            plt.plot(EXP_history[state][0][int(0.2*iterations):])
        plt.title("EXP of " + agent.actions[0] + " when " + state + " in " + decision_problem.description)
        plt.show(decision_problem.description + state + " EXP of " + agent.actions[0])
        #plt.savefig("C:\Users\Joar\Desktop\Decision Problems\plot "+decision_problem.description + state + " EXP of " + agent.actions[0] + ".png")

        for distribution_history in distribution_histories:
            plt.plot(EXP_history[state][1][int(0.2*iterations):])
        plt.title("EXP of " + agent.actions[1] + " when " + state + " in " + decision_problem.description)
        plt.show(decision_problem.description + state + " EXP of " + agent.actions[1])
        #plt.savefig("C:\Users\Joar\Desktop\Decision Problems\plot "+ decision_problem.description + state + " EXP of " + agent.actions[1] + ".png")

    print("###")
