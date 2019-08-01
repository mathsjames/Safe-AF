import numpy as np
from Decision_Problems.prisoners_dilemma_against_copy import Prisoners_Dilemma_against_copy

class customisable_prisoners_dilemma(Prisoners_Dilemma_against_copy):
    def __init__(self, ratio):
        Prisoners_Dilemma_against_copy.__init__(self)
        self.utility_dict = {"cooperate+copy cooperates":1000,
                             "cooperate+copy defects":0,
                             "defect+copy coperates":1000+int(1000*ratio),
                             "defect+copy defects":int(1000*ratio)}
