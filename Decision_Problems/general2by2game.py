import numpy as np
from Decision_Problems.decision_problem import Decision_Problem

class General2by2(Decision_Problem):

    # A general implementation of 2 by 2 problems such as PD vs self and EB only looking at cases where blackmail is sent

    def __init__(self, rewards, response_distribution, name):
        self.actions = ["action1", "action2"]

        self.states = ["response1", "response2",
                       "action1+response1", "action1+response2", "action2+response1", "action2+response2"]

        self.end_states = ["action1+response1", "action1+response2", "action2+response1", "action2+response2"]

        self.epistemic_states = ["NewRound", "END"]

        self.state_to_epistemic_state_dict = {"response1":"NewRound",
                                              "response2":"NewRound",
                                              "action1+response1":"END",
                                              "action1+response2":"END",
                                              "action2+response1":"END",
                                              "action2+response2":"END",
                                             }

        self.causation_dict = {("response1", "action1"):"action1+response1",
                               ("response2", "action1"):"action1+response2",
                               ("response1", "action2"):"action2+response1",
                               ("response2", "action2"):"action2+response2",
                              }

        self.description = "2by2 Game: "+name

        self.response_distribution=response_distribution

        self.rewards=rewards

    def reset_with(self, agent):
        self.finished = False
        action_distribution = agent.get_action_distribution("NewRound")
        response = np.random.choice(["response1","response2"], 1, p=self.response_distribution(action_distribution))[0]
        self.state = response
        rewards=self.rewards(action_distribution);
        self.utility_dict = {"action1+response1":rewards[0][0],
                             "action1+response2":rewards[0][1],
                             "action2+response1":rewards[1][0],
                             "action2+response2":rewards[1][1],
                            }
