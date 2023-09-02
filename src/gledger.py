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
      
    def read_data(self, path:str):
        """
        Reads in an xlxs (or csv) file or a directory containing xlxs (or csv) files and returns a single DataFrame.
        
        Parameters
        ----------
        path: str 
            Path to a raw general ledger file or directory 
            
        Returns
        -------
            A single dataframe sorted by company code and year
        """
        complete_df = pd.DataFrame
        
        if os.path.isdir(path):
            file_names = [path+file for file in os.listdir(path) if (file.endswith('.csv') or file.endswith('.xlsx'))]
            
            assert len(file_names) > 0, "There do not seem to be any data files in your directory"
            
            df_list = [] 
            for file in file_names:
                if file.endswith(".xlsx"):
                    xls = pd.ExcelFile(file)
                    current_df = xls.parse('Sheet1', header=1)
                else: 
                    current_df = pd.read_csv(file)
                df_list.append(current_df)
            complete_df = pd.concat(df_list)
            
        
        elif os.path.isfile(path):
            if path.endswith('.xlsx'):
                xls = pd.ExcelFile(path)
                complete_df = xls.parse('Sheet1', header=1)
            elif path.endswith('.csv'):
                complete_df = pd.read_csv(path)
        
        
        complete_df['$'] = complete_df['$'].apply(utils.convert_to_float)
        complete_df.replace('#', 'n/a', inplace=True)
        return complete_df.sort_values(by=["Company Code", "Fiscal Year"])
    

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


    def agg_lines(self, df:pd.DataFrame, single_cols=['Transaction Code', 'JE Doc Type', 'JE Created By'], agg_cols=['G/L Account']):
        """
        Aggregates Transaction lines by joining on Company Code + JE DOC # + Fiscal Year, 
        recording agg_cols columns as sets and $ amount as net change over the transaction.  

        Note: Will perform a pd.join with all columns in cols, so care must be taken to only include cols that are homogeneous across transaction lines.

        Parameters:
        -----------

        df: pd.DataFrame 
            A pandas dataframe representation of a General Ledger 
        
        single_cols: list 
            A list of columns to be included in the complete transaction data set that are homogenous 
            across transaction lines.


        agg_cols: list 
            A list of columns to be recorded as a set across transacation lines 
        """
        # Subset DataFrame
        sub_df = df.copy()[single_cols + agg_cols + ['Company Code', 'JE Doc #', 'Fiscal Year', '$']]

        # Convert Non-Set datatypes to strings
        sub_df[single_cols] = sub_df[single_cols].astype(str)
        
        # Create Transaction ID column 
        sub_df['ID'] = df['Company Code'].astype(str) + df['JE Doc #'].astype(str) + df['Fiscal Year'].astype(str)
        

        # Track total amount across transaction 
        sub_df['$'] = sub_df['$'].apply(utils.abs_amount)
        
        single_agg = sub_df.drop_duplicates(subset=['ID'])[['ID'] + single_cols]

        set_agg = sub_df.groupby('ID')[agg_cols].agg(set).reset_index()
        amount_agg = sub_df.groupby('ID')['$'].sum().reset_index() 

        agg = pd.merge(single_agg, set_agg, on='ID')
        agg = pd.merge(agg, amount_agg, on='ID')
        
        return  agg
    


    def bar(self, df:pd.DataFrame, col='G/L Account', threshold=100): 
        
        #TODO: 
        # Only apply tuple to set types. Currently not working as is. 
        
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

    
    
    def fit(): 
        # STUB! 
        return 
    
    def fit_transform(): 
        # STUB! 
        return 

    


    



        

        



        

