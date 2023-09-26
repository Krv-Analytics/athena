
import os 
import pandas as pd
import numpy as np
import yaml 
import pickle
import shutil

from azure.storage.blob import (
BlobServiceClient,
ContainerClient,
BlobClient
)

def read_data(path:str):
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
        
        
        complete_df['$'] = complete_df['$'].apply(convert_to_float)
        complete_df.replace('#', 'n/a', inplace=True)
        return complete_df.sort_values(by=["Company Code", "Fiscal Year"])



def agg_lines(df:pd.DataFrame, single_cols=['Transaction Code', 'JE Doc Type', 'JE Created By'], agg_cols=['G/L Account']):
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

    Returns
    -------
    A dataframe of aggregated transactions 
    """
    # Subset DataFrame
    sub_df = df.copy()[single_cols + agg_cols + ['Company Code', 'JE Doc #', 'Fiscal Year', '$']]

    # Convert Non-Set datatypes to strings
    sub_df[single_cols] = sub_df[single_cols].astype(str)
    
    # Create Transaction ID column 
    sub_df['ID'] = df['Company Code'].astype(str) + df['JE Doc #'].astype(str) + df['Fiscal Year'].astype(str)
    sub_df['ID'] = sub_df['ID'].astype(int)

    # Track total amount across transaction 
    sub_df['$'] = sub_df['$'].apply(abs_amount)
    
    single_agg = sub_df.drop_duplicates(subset=['ID'])[['ID'] + single_cols]

    set_agg = sub_df.groupby('ID')[agg_cols].agg(set).reset_index()
    amount_agg = sub_df.groupby('ID')['$'].sum().reset_index() 

    agg = pd.merge(single_agg, set_agg, on='ID')
    agg = pd.merge(agg, amount_agg, on='ID')

    # TODO: Sort by Fiscal Year, then JE Doc #, then Company Code in 'ID' 
    
    return  agg


def find_lines_from_transaction(index: int, t_agg: pd.DataFrame, t_lines: pd.DataFrame): 
    """
    Returns the dataframe subset of t_lines that corresponds to transaction lines of the
    specified index in t_agg. 
    """
    # Pull transaction ID from index
    id = str(t_agg.iloc[index]['ID'])

    # Breakdown ID into Company Code, JE DOC #, and Fiscal Year 
    cc = int(id[:4])
    doc_num = int(id[4:-4])
    year = int(id[-4:])



    # Return subset of transaction lines corresponding to transaction
    return t_lines[(t_lines['Company Code'].astype(int) == cc) 
                            & (t_lines['JE Doc #'].astype(int) == doc_num) 
                            & (t_lines['Fiscal Year'].astype(int) == year)]


def get_transaction_sample(t_agg: pd.DataFrame, t_lines:pd.DataFrame, num_samples=100): 
    """
    Returns an aggregated transaction dataframe sample and corresponding t_lines by randomly sampling from 
    t_agg and rebuilding a lines dataframe corresponding to the sample. 

    """

    sampled_agg = t_agg.sample(num_samples)

    # Apply the function to each index in 'sampled_data' and store the results in a list
    subset_list = sampled_agg.index.map(lambda index: find_lines_from_transaction(index, t_agg, t_lines)).tolist()

    # Concatenate all the subsets into a single DataFrame
    sampled_lines = pd.concat(subset_list, ignore_index=True)
    sampled_agg.reset_index(drop=True, inplace=True)
    sampled_lines.reset_index(drop=True, inplace=True)
    return sampled_agg, sampled_lines

    

def convert_to_float(value):
    try:
        return float(value.replace(',', ''))
    except:
        return value
    
def abs_amount(id):
    if id <0:
        return 0
    else:
        return id
    


def get_transaction_subset(indices, t_lines, t_agg): 
    """
    Returns the subset of t_agg according to indices and corresponding t_lines dataframe. 

    """

    agg_subset = t_agg.iloc[indices]
    subset_list = agg_subset.index.map(lambda index: find_lines_from_transaction(index, t_agg, t_lines)).tolist()
    lines_subset = pd.concat(subset_list, ignore_index=True)
    agg_subset.reset_index(drop=True, inplace=True)
    lines_subset.reset_index(drop=True, inplace=True)

    return agg_subset, lines_subset


def write_to_thema(thema_path, run_name, agg_subset, lines_subset, lens_dict, config_file): 
    """
    Writes aggregated transaction dataframe and transaction line dataframe into 
    *thema_path*/data/*run_name*/raw/
    """
    
    target_raw_dir = os.path.join(thema_path, 'data', run_name, 'raw')
    target_clean_dir = os.path.join(thema_path, 'data', run_name, 'clean')
    target_config_file = os.path.join(thema_path, 'data', run_name, 'config.yaml')


    # Check if thema_path exists
    if not os.path.exists(thema_path):
        raise Exception(f"The specified path to your THEMA repository does not exist.")
    
    # Check if the target directories exist, if not, create them
    if not os.path.exists(target_raw_dir):
        os.makedirs(target_raw_dir)
    if not os.path.exists(target_clean_dir):
        os.makedirs(target_clean_dir)

    # Define the file paths for the dataframes
    t_agg_path = os.path.join(target_raw_dir, 't_agg.pkl')
    t_lines_path = os.path.join(target_raw_dir, 't_lines.pkl')

    lens_path = os.path.join(target_clean_dir, 'lens.pkl')

    # Write the dataframes to the specified paths
    agg_subset.to_pickle(t_agg_path)
    lines_subset.to_pickle(t_lines_path)

    # write lens to clean data path 
    with open(lens_path, 'wb') as f:
        pickle.dump(lens_dict, f)

    # write config file 
    shutil.copy(config_file, target_config_file)



def subset_lines(subset:dict, lines:pd.DataFrame): 
    
    """
    Will loop through the subset dictionary, which has keys that correspond to columns in 
    lines, and values that the column will be subset to. 

    e.g. subset = {Fiscal Year: 2021} will subset the dataframe but taking all rows 
    with column field "Fiscal Year" equal to 2021. 

    This function returns both the subset of lines, as well as its corresponding aggregrated 
    columns dataframe.

    """


    subset_lines = lines.copy()  # Create a copy of the original DataFrame to avoid modifying it.

    for column, value in subset.items():
        # Filter the DataFrame based on the specified column and value.
        subset_lines = subset_lines[subset_lines[column] == value]

    subset_lines.reset_index(inplace=True, drop=True)

    subset_agg = agg_lines(subset_lines)

    return subset_agg, subset_lines




