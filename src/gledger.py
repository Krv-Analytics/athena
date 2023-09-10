# GLedger Class 

import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import utils

class GLedger: 
    """
        A class designed to faciliate the interaction and analysis of transaction 
        line and transaction data when receiving xlxs (or csv) General Ledger files. 
        This class supports some basic statiscal analysis and functions for fitting and generating 
        projection lenses. 


        Members
        -------

        flow_directions: dict()



        Member Functions
        ----------------
    """

    ################################################################################################
    #
    #   Initilization
    #
    ################################################################################################
    
    def __init__(self, t_lines=None, t_agg=None):
        """
        Intitializes GLedger class from a variety of build stages, supporting 
        xlsx concatenation and conversion as well as line aggregation in flight for 
        preparation towards pipelined capabilites with `krv_mapper`. 

        t_lines: pd.DataFrame, str

            Options for t_lines 

            (1) an already compiled dataframe containing at the minimum columns 'Company Code', 'JE Doc #' 
            'Fiscal Year', 'G/L Account' and '$'. 

            (2) a path to a pickle file containing such a dataframe as described above. 

            (3) a path to an xlsx file or folder, which will be compiled into a dataframe 
            using the read_data function in utils. 
        
        t_agg: None, pd.DataFrame, str 

            Options for t_agg 

            (1) None - t_agg will be compiled from t_lines using agg_lines function from utils 

            (2) pd.DataFrame - a pandas dataframe containing columns 'ID', 'G/L Account' and '$' corresponding 
            to the passed t_lines object 

            (3) str - a path to a pickle file containing such a dataframe. 

        """
        
    # Handling t_lines initialization
        assert t_lines is not None, "Please specifiy a transaction line data set with t_lines"

        # Case 1:  t_lines is a path
        if type(t_lines) == str: 
            if t_lines.endswith('.pkl'):
                self._t_lines = pd.read_pickle(t_lines)
            elif t_lines.endswith('.xlsx'):
                self._t_lines = utils.read_data(t_lines) 
            else: 
                assert 1 == 0, "Only .pkl and .xlsx files are supported at this time."
        
        # Case 2: t_lines is dataframe 
        elif type(t_lines) == pd.DataFrame: 
                self._t_lines = t_lines 
        
        # Unsupported data type for t_lines 
        else: 
            assert 1 == 0, "Unsupported data type passed to t_lines."

    # Handling t_agg intialization 

        # Case (1): t_agg is None 
        if t_agg is None: 
            self._t_agg = utils.agg_lines(self._t_lines) 
        
        # Case (2): t_agg is path 
        elif type(t_agg) == str: 
            assert t_agg.endswith('.pkl'), "Only pickle files are supported at this time."
            self._t_agg = pd.read_pickle(t_agg)
        
        # Case (3): t_agg is dataframe
        elif type(t_agg) == pd.DataFrame:
            self._t_agg = t_agg 
        
        # Unsuported
        else:
            assert 1==0, "Unsupported data type passed to t_agg."
        
        
        # Data members used for fitting
        self.flow_directions = None
        self._projection = None 


    
    ################################################################################################
    #
    #   Getters 
    #
    ################################################################################################


    def get_tlines_df(self):
        """
        Returns dataframe of transaction lines. 
        """
        return self._t_lines
    
    def get_agg_df(self):
        """
        Returns dataframe of aggregated transactions. 
        """
        return self._t_agg
    

    def get_transaction_from_line(self, index):
        """
        Given an index of a transaction line, will return the aggregated transaction. 

        Parameters
        ----------

        index: int 
            Inded of a transaction line in t_lines
        """

        # Extract line from transaction line df 
        line = self._t_lines.iloc[index]
        
        # Concatenate columns fields to construct transaction ID
        id = line['Company Code'].astype(str) + line['JE Doc #'].astype(str) + line['Fiscal Year'].astype(str)

        # Return aggregated transaction
        return self._t_agg[self._t_agg['ID'] == id]
    

    def get_lines_from_transaction(self, index): 
        """
        Given an index of a transaction, will return the associated transaction lines. 

        Parameters
        ----------
        index: int 
            Inded of a transaction in t_agg        
        """
        # Pull transaction ID from index
        id = str(self._t_agg.iloc[index]['ID'])

        # Breakdown ID into Company Code, JE DOC #, and Fiscal Year 
        cc = int(id[:4])
        doc_num = int(id[4:-4])
        year = int(id[-4:])

        # Return subset of transaction lines corresponding to transaction
        return self._t_lines[(self._t_lines['Company Code'] == cc) 
                              & (self._t_lines['JE Doc #'] == doc_num) 
                              & (self._t_lines['Fiscal Year'] == year)]
    


    ################################################################################################
    #
    #   Fitting 
    #
    ################################################################################################

    
    def fit(self): 
        """
        Computes unitary and pairwise interactions of G/L Accounts to determine common account flow 
        directions. Set the `flow_directions` dictionary with keys being tuples of G/L Accounts and values 
        being a length 4 array. 
                (gl1,gl2)  ->   (++, +-, -+, --)

                ++: number of transactions containing gl1 and gl2 where both gl1 and gl2 
                recieve 
                +-: " " where gl1 recieves money and gl2 deposits money 
                -+: " " where gl1 deposits and gl2 receives 
                -- " " where both gl1 and gl2 deposit 

        Note: Key pairs are in sorted order!
        """


        # Generating pairwise G/L Account list 
        unique_accounts = sorted(self._t_lines['G/L Account'].unique()) 
        
        # Initializing flow_directions dictionary 
        self.flow_directions = {key:np.zeros(3) for key in unique_accounts}
        
        
        for key in unique_accounts: 
                
                # Recieve and Deposit 
                R = 0 
                D = 0 

                transaction_list = self._t_agg[self._t_agg['G/L Account'].apply(lambda x: key in x)].index
                  # Get transaction lines from transaction 
                for transaction_index in transaction_list:
                    lines = self.get_lines_from_transaction(transaction_index) 
                    flow = lines[lines['G/L Account'] == key].reset_index().iloc[0]['$']
                    # Account Receives
                    if flow > 0: 
                        R = R + 1 
                    # Account Deposits
                    else: 
                        D = D + 1 
                
                total = R + D  
                self.flow_directions[key][2] = total 
                self.flow_directions[key][0] = R/ total 
                self.flow_directions[key][1] = D/total 


        account_pairs = []
        for i in range(0,len(unique_accounts)):
            for j in range(i+1, len(unique_accounts)): 
                account_pairs.append((unique_accounts[i], unique_accounts[j]))
         
        # Compute flows for each pair 
        for key_pair in account_pairs: 
            # Get list of transactions containing G/L Account Pair 
            transaction_list = self._t_agg[self._t_agg['G/L Account'].apply(lambda x: (key_pair[0] in x) & (key_pair[1] in x))].index
            
            # RecieveRecieve, RecieveDeposit, DepositRecieve, DepositDeposit
            RR = 0 
            RD = 0 
            DR = 0 
            DD = 0 

            # Get transaction lines from transaction 
            for transaction_index in transaction_list:
                lines = self.get_lines_from_transaction(transaction_index) 
                flow1 = lines[lines['G/L Account'] == key_pair[0]].reset_index().iloc[0]['$']
                flow2 = lines[lines['G/L Account'] == key_pair[1]].reset_index().iloc[0]['$']

                # Split into 4 cases

                # Both recieve
                if flow1 > 0 and flow2 > 0: 
                    RR = RR + 1 
                
                # First recieves, second deposits 
                elif flow1  > 0 and flow2 < 0: 
                    RD = RD + 1
                
                # First deposits, second recieves
                elif flow1  < 0 and flow2 > 0: 
                    DR = DR + 1
                
                # Both desposit 
                else: 
                    DD = DD + 1
            
            total =(RR + RD  + DR + DD)
            if total > 0:
                self.flow_directions[key_pair] = np.zeros(5)
                self.flow_directions[key_pair][4] = total 
                self.flow_directions[key_pair][0] = RR/total 
                self.flow_directions[key_pair][1] = RD/total 
                self.flow_directions[key_pair][2] = DR/total 
                self.flow_directions[key_pair][3] = DD/total 
    

    
    def fit_transform(self, transformation='flow', df=None): 
        """
        
        Parameters
        ----------

        transformation: str
                Supported types are 'flow', 'interaction_flow', 'frequency', and 'interaction_frequency'

        df: pd.DataFrame 
                Data set to perform transformation on (dafault is None, in which case 
                transformation for t_agg is returned. )
        """
        
        
        # Run Fitting Function if unfitted
        if self.flow_directions is None and transformation != 'frequnecy': 
            self.fit() 

        # May pass new df after training on t_agg
        if df is None: 
            df = self._t_agg

        assert transformation in ['flow', 'interaction_flow', 'frequency', 'interaction_frequency']

        if transformation == 'flow': 
            return df.apply(self.flow_1Ddescription,axis=1)
            
        
        
        if transformation == 'frequency': 
            gl_counts =  df['G/L Account'].apply(frozenset).value_counts().to_dict()
            return df['G/L Account'].apply(lambda x: gl_counts[frozenset(x)])

        
        else: 
            print('Other transformations are not supported at this time.')
    




    def flow_1Ddescription(self, row): 
        """
        Helper function for determining normalcy of flow directions within a 
        G/L Set. 

        Parameters
        ----------
        row: row of a pandas dataframe 
        
        """
        # Array to be return as a projection
        line_df = self.get_lines_from_transaction(row.name)
        account_set = row['G/L Account']
        
        
        # Compute transaction flow from set of G/L Accounts 
        transaction_score = 0 
        for account in account_set:
            account_df = line_df[line_df['G/L Account'] == account].reset_index()
            if(account_df.iloc[0]['$'] > 0):
                transaction_score += self.flow_directions[account][0]
            else: 
                transaction_score += self.flow_directions[account][1]
        
        # Return Normalized score  
        return transaction_score/len(account_set)




    def flow_2Ddescription(self, gl_set: set, norm='l1'): 
        """
        Helper function for determining normalcy of two-way flow interactions. 
        
        Parameters
        ----------
        gl_set: set 
            A Set of G/L Accounts. 
        
        norm: string 
            l1 or l_inf 
        """
        
        return 





    

    ################################################################################################
    #
    #   Visualizations  
    #
    ################################################################################################


    
    def bar(self,  col='G/L Account', threshold=0): 
        """
        Plots a bar graph of a given columns frequencies. 

        Parameters 
        ----------
            col: 
                The column to perform the frequency analysis and plotting of 
            
            threshold: int 
                A cut-off point under which all occurences with smaller frequencies are 
                removed from the plot.  
        """
        
        if self._t_agg[col].dtype == set:
            counts = self._t_agg[col].apply(tuple).value_counts()
        else:
            counts = self._t_agg[col].value_counts()
        filtered_counts = counts[counts > threshold]

            # Plot a bar plot
        plt.figure(figsize=(10, 6))
        filtered_counts.plot(kind='bar')
        plt.xlabel(f'{col}')
        plt.ylabel('Count')
        plt.title(f'{col} Counts (Count > {threshold})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        print(f"Percentage of Transactions with > {threshold} occurences of {col}: {round(sum(filtered_counts)/len(self._t_agg)*100, 2)}%")
        plt.show()



    
    def scatter(self): 
        """
        STUB!! 

        Scatter plot distribution of projection. (Support a couple of distribution methods),
        - 3D 
        - UMAP 2D 
        
        """
        return  
    


    



        

        



        

