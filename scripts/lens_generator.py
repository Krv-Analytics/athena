import os 
import sys 
import pandas as pd 
import psutil
import time
import threading
from pympler import asizeof

from dotenv import load_dotenv
load_dotenv() 
root = os.getenv('root')
config_file = os.getenv('config_file')
thema_path = os.getenv('THEMA')

sys.path.append(root+'/src/')
from gledger import GLedger
import utils
import config_utils

config_data = config_utils.load_config(config_file)

# Setting Lens Configuration Variables 
contamination = config_data['contamination']

transaction_line_file = root + "/" +  config_data["tlines_path"]
transaction_agg_file = root + "/" + config_data["tagg_path"]

def monitor_memory_usage(stop_event):
    process = psutil.Process()
    while not stop_event.is_set():
        memory_info = process.memory_info()
        sys.stdout.write(f"\rMemory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        sys.stdout.flush()
        time.sleep(2)


def gledger_mem_breakdown(gledger):
    total_size = asizeof.asizeof(gledger)
    t_agg_size = asizeof.asizeof(gledger._t_agg)
    t_lines_size = asizeof.asizeof(gledger._t_lines)
    flow_directions_size = asizeof.asizeof(gledger.flow_directions)

    print("\n\nGLedger Memory Breakdown:")
    print(f"Total Ledger Size: {total_size / 1024 / 1024:.2f} MB")
    print(f"t_agg Size: {t_agg_size / 1024 / 1024:.2f} MB")
    print(f"t_lines Size: {t_lines_size / 1024 / 1024:.2f} MB")
    print(f"flow_directions Size: {flow_directions_size / 1024 / 1024:.2f} MB")



if __name__ == "__main__":

    start_time = time.time()

    # Create an event to signal the monitoring thread to stop
    stop_event = threading.Event()

    # Start monitoring memory usage in a separate thread
    monitoring_thread = threading.Thread(target=monitor_memory_usage, args=(stop_event,))
    monitoring_thread.start()

    try:
        # Instantiate Class 
        gledger = GLedger(t_agg=transaction_agg_file, t_lines=transaction_line_file)

        agg_time = time.time() - start_time
        fit_start = time.time()
        
        # Run your GLedger's fit method
        gledger.fit()
        # Time to fit 
        fit_time = time.time() - fit_start

        projection_start = time.time()

        lens_dict = gledger.get_lens(contamination=contamination, sample_norm='l2')
        assert lens_dict['clean_data'] is not None 

        # Write data to thema 
        agg_subset, lines_subset = utils.get_transaction_subset(lens_dict['hyperparameters'], t_lines = gledger.get_tlines_df(), t_agg = gledger.get_agg_df())
        utils.write_to_thema(thema_path, config_data['run_name'], agg_subset, lines_subset, lens_dict, config_file)

        # Projection Time 
        projection_time = time.time() - projection_start

        # Total Time 
        end_time = time.time() - start_time
    except Exception as e:
        stop_event.set()
        monitoring_thread.join()
        print(f"\n\nScript Exited Unsuccessfully with error: {e}")
        exit()

    # Wait for the monitoring thread to finish
    stop_event.set()
    monitoring_thread.join()

   # Runtime Specs 
   
    gledger_mem_breakdown(gledger)
    print(f"\n\nRuntime specs for {len(gledger._t_agg)} transactions \n")
    print(f"Aggregation time: {agg_time}")
    print(f"Fitting time: {fit_time}")
    print(f"Projection time: {projection_time}")
    print(f"Total Runtime: {end_time}")

