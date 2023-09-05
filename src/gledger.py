# GLedger Class 

import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import utils

class GLedger: 
    """
        A class designed to faciliate the transition from transaction lines to transactions when 
        receiving xlxs (or csv) General Ledger files. Only with general housekeeping, this class 
        supports some basic statiscal analysis and functions for generating projection lenses. 
    """
    
    def __init__(self, t_lines=None, t_agg=None):
        """
        Intitializes GLedger class from a variety of build stages, supporting 
        xlsx concatenation and conversion as well as line aggregation in flight for 
        preparation towards pipelined capabilites with `krv_mapper`. 

        t_lines: None, pd.DataFrame, str 
            Passing None (ignores data) allows for use of read_data and agg_lines functionality for developement 
            and testing when initlizing as None 

            May pass an already compiled dataframe containing at the minimum columns 'Company Code', 'JE Doc #' 
            'Fiscal Year', 'G/L Account' and '$'. 

            May pass a path to a pickle file containing such a dataframe as described above. 

            May pass a path to an xlsx file or folder, which will be compiled into a dataframe 
            using the read_data function. 
        
        t_agg: None, pd.DataFrame, str 

            Will be treated as `None` if a t_lines argument is not passed. 

            If left `None`, t_agg will be computed from t_lines. 



        """
        if t_lines is not None: 
            self._t_lines = pd.read_pickle()
            self._t_agg = self.agg_lines(self._t_lines) 
        self.flow_directions = None 
    


    def clean_dtypes(self, df:pd.DataFrame):
        """
        STUB!!
        Summarizes data types and converts all objects to string representations. DateTime data columns are converted to form two new columns, namely

        date: YYYYMMDD 
        time: HHMMSS.

    
        Note: It is advised to handle data type issues manually if interested in passing column to agg_lines. 

        Parameters
        ----------
        df: pd.DataFrame 

        """

        # STUB! 
        return df

    
    def bar(self, df:pd.DataFrame, col='G/L Account', threshold=0): 
        """
        Plots a bar graph of a given columns frequencies. 

        Parameters 
        ----------
            df: pd.DataFrame 
                A dataframe of transaction (or transaction line) information 
            col: 
                The column to perform the frequency analysis and plotting of 
            
            threshold: int 
                A cut-off point under which all occurences with smaller frequencies are 
                removed from the plot.  
        """
        
        if df[col].dtype == set:
            counts = df[col].apply(tuple).value_counts()
        else:
            counts = df[col].value_counts()
        filtered_counts = counts[counts > threshold]

            # Plot a bar plot
        plt.figure(figsize=(10, 6))
        filtered_counts.plot(kind='bar')
        plt.xlabel(f'{col}')
        plt.ylabel('Count')
        plt.title(f'{col} Counts (Count > {threshold})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        print(f"Percentage of Transactions with > {threshold} occurences of {col}: {round(sum(filtered_counts)/len(df)*100, 4)}%")
        plt.show()

    

    def fit(self, t_lines:pd.DataFrame, t_agg:pd.DataFrame): 
        """
        Computes pairwise interactions of G/L Accounts to determine common account flow 
        directions. Creates a dictionary with keys being tuples of G/L Accounts and values 
        being a length 4 array. 
                (gl1,gl2)  ->   (++, +-, -+, --)

                ++: number of transactions containing gl1 and gl2 where both gl1 and gl2 
                recieve 
                +-: " " where gl1 recieves money and gl2 deposits money 
                -+: " " where gl1 deposits and gl2 receives 
                -- " " where both gl1 and gl2 deposit 

        Parameters 
        ----------
        t_lines: pd.DataFrame
            A dataframe of transaction line information. Must have the columns 
        'G/L Account' and '$'. 

        t_agg: pd.DataFrame 
            A dataframe of transactions containing a column 'G/L Account' that holds 
            sets of G/L Accounts 
        """
        unique_accounts = sorted(t_lines['G/L Account'].unique()) 
        account_pairs = []
        for i in range(0,len(unique_accounts)):
            for j in range(i+1, len(unique_accounts)): 
                account_pairs.append(np.pair(unique_accounts[i], unique_accounts[j]))
        
        self.flow_directions = dict(account_pairs)

        # for key_pair in self.flow_directions.keys(): 
    


    
    def fit_transform(): 
        # STUB! 
        return 
    

    def scatter():
        # STUB! 
        return 

    


    



        

        



        

