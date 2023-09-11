import os 
import pandas as pd 
import numpy as np
from itertools import combinations

from gledger import GLedger


class Athena: 
    """
    Class for creating ML friendly representations of General Ledger data equipped with 
    labels from `krv_mapper`. A natural extention of GLedger, this class will procure
    artificially labelled data and interface with pytorch to train an MLP classifier of 
    anomalous trainsactions. 

    Members
    -------


    Member Functions 
    ---------------- 
    
    
    """

    def __init__(self, gledger:GLedger, labels:str):
        """
        In Developement! 

        Gledger member assinged at intialization. Imagining that labels will 
        be a file outputed from `krv_mapper`. 


        """
        
        self.gledger = gledger
        self.labels = labels


    
    def get_transaction_footprint(self,t_lines:pd.DataFrame): 
        """
        Represents a single transaction composed of multiple lines into a matrix 
        of flow directions. This footprint representation will be used to train a classifier
        of transactions when supplied with labels. 


        Parameters
        ----------

        t_lines: pd.DataFrame 
            A Pandas dataframe containing the columns 'G/L Account' and '$' with 
            the same 'Company Code' + 'JE Doc #' + 'Fiscal Year' (ie transaction ID). 
            This df corresponds to the transaction lines of a single transaction. 
        
        
        e.g.  Consider a General Ledger consisting of transactions with G/L Accounts 
        in the set {A, B, C, D, E, F, G}. Note that G/L accounts will apear in sorted order.  

        Transaction T given by lines:  


            JE Doc #    |    G/L Account     |   $ 
        0    001        |        A           |   +12 
        1    001        |        C           |   - 3 
        2    001        |        E           |   -14
        3    001        |        F           |   + 5 
                

        FootPrint of T (before scaling): 
        
        
                A           B       C           D          E         F       G 
            ___________________________________________________________________
        
        A   | (12,0,0,0)   0   (0,15,0,0)      0    (0,26,0,0)(17,0,0,0)     0
            |
        B   |    0          0       0           0         0          0       0
            |
        C   | (0,15,0,0)    0    (0,3,0,0)      0     (0,0,0,17) (0,0,8,0)   0        
            |
        D   |    0          0        0          0          0         0       0               
            |
        E   | (0,26,0,0)    0    (0,0,0,17)     0      (0,14,0,0) (0,0,19,0) 0   
            |
        F   | (+17,0,0,0)   0    (0,0,8,0)      0       (0,0,19,0) (5,0,0,0) 0 
            |
        G   |    0          0        0          0            0         0     0
            ___________________________________________________________
        
        
        """
        # initialize matrix 
        n = len(self.gledger.unique_accounts)
        index_dict = {id_value: index for index, id_value in enumerate(self.gledger.unique_accounts)} 
        footprint_matrix = np.zeros((n,n,4))

        amounts = t_lines['$'].to_list()
        g_accounts = t_lines['G/L Account'].to_list()

        # set the diagonal elements 
        for i in range(len(amounts)): 
            k = 0
            if amounts[i] < 0: 
                k = 1
            footprint_matrix[index_dict[g_accounts[i]], index_dict[g_accounts[i]], k] = 1 # abs(amounts[i])


        # Loop through unique pairs: 
        unique_pairs = list(combinations(t_lines['G/L Account'].unique(),2))

        for key_pair in unique_pairs:
            amount_1 = t_lines[t_lines['G/L Account'] == key_pair[0]].reset_index().iloc[0]['$']
            amount_2 = t_lines[t_lines['G/L Account'] == key_pair[1]].reset_index().iloc[0]['$']

            if amount_1 > 0 and amount_2 > 0: 
                k=0 
            elif amount_1 > 0 and amount_2 < 0:
                k=1
            elif amount_1 < 0 and amount_2 > 0:
                k=2 
            else: 
                k=3
            footprint_matrix[index_dict[key_pair[0]],index_dict[key_pair[1]], k] = 1# abs(amount_1) + abs(amount_2)
            footprint_matrix[index_dict[key_pair[1]],index_dict[key_pair[0]], k] = 1 #abs(amount_1) + abs(amount_2)
        
        return footprint_matrix
            








