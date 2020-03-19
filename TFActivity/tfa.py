import numpy as np
import pandas as pd
from scipy import linalg

EXPRESSION = pd.read_csv("../expression.tsv", sep='\t', index_col=0)
PRIORS = pd.read_csv("../gold_standard.tsv", sep='\t', index_col=0)

class TFA:
    """
    TFA calculates Transcription Factor Activity using Local Network Component analysis
    ref: https://www.sciencedirect.com/science/article/pii/S1046202317300506?via=ihub

    Parameters
    --------
    prior: pd.dataframe
        binary or numeric g by t matrix stating existence of gene-TF interactions.
        g: gene, t: TF.

    expression_matrix: pd.dataframe
        normalized expression g by c matrix. g--gene, c--conditions
    """

    def __init__(self, expression_matrix=EXPRESSION, priors_data=PRIORS):
        self.expression_matrix = expression_matrix
        self.priors_data = priors_data

    def local_network_component_analysis(self):
        pass

    def fast_network_component_analysis(self):
        pass

    def align_transcription_factor_activation(self):
        pass

    def optimize_weights(self):
        pass
