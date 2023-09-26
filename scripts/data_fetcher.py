import os 
import sys 
import pandas as pd 
import psutil
import time
import threading
from pympler import asizeof

from azure.storage.blob import BlobClient

from dotenv import load_dotenv
load_dotenv() 
root = os.getenv('root')
config_file = os.getenv('config_file')
thema_path = os.getenv('THEMA')
az_connection = os.getenv('azure_connection_string')

sys.path.append(root+'/src/')
from gledger import GLedger
import utils
import config_utils

config_data = config_utils.load_config(config_file)

# Setting Data Configuration Variables 
sample_size = config_data['sample_size']
subset = config_data["subset"]
container = config_data["azure_container"]
agg_blob_name = "agg_" + config_data["data_batch"]
lines_blob_name = "lines_" + config_data["data_batch"]


# Configure Azure Client 
agg_blob = BlobClient.from_connection_string(conn_str=az_connection, container_name=container, blob_name=agg_blob_name)
lines_blob = BlobClient.from_connection_string(conn_str=az_connection, container_name=container, blob_name=lines_blob_name)


if __name__ == "__main__":
    
    dataset_line_file = root + "/datasets/" + config_data['azure_container'] + "_" + lines_blob_name + '.pkl'
    dataset_agg_file = root + "/datasets/" + config_data['azure_container']+ "_" + agg_blob_name + '.pkl'

    # Make sure datasets directory exists
    if not os.path.exists(root + "/datasets/"):
        os.makedirs(root + "/datasets/")

    # Load Aggregated Transaction Dataset 
    if not os.path.isfile(dataset_agg_file): 
        with open(dataset_agg_file, 'wb') as f: 
            blob_data = agg_blob.download_blob()
            blob_data.readinto(f)
    else: 
        print("Aggregated DataSet already exists locally.")
    
    # Load Transaction lines dataset 
    if not os.path.isfile(dataset_line_file): 
        with open(dataset_line_file, 'wb') as f: 
            blob_data = lines_blob.download_blob()
            blob_data.readinto(f)
    else: 
        print("Lines DataSet already exists locally.")

    # Read in Dataframe from local file  
    lines = pd.read_pickle(dataset_line_file)
    agg = pd.read_pickle(dataset_agg_file)

    # Subset dataframe 
    if not subset == "None" : 
        agg, lines = utils.subset_lines(subset, lines)
    

    assert len(agg) > 0, "You appear to have generated a subset of length 0. Please \
        make sure you are subsetting the lines dataframe correctly in the config.  \
        A common bug is mismatched data types between the df and config file."


    if type(sample_size) == int and sample_size > 0:
        agg, lines = utils.get_transaction_sample(t_agg=agg, t_lines=lines, num_samples=sample_size)


    # Create Run Name Dir in Data 
    if not os.path.exists(root + "/data/" + config_data['run_name']):
        os.makedirs(root + "/data/" + config_data['run_name'])

    # Write the subset files 
    lines.to_pickle(root + config_data['tlines_path']) 
    agg.to_pickle(root + config_data['tagg_path'])


    print("Successfully written your transaction sample.")



        
    


