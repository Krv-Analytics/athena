import sys 
import pandas as pd 
import pickle 
import psutil
import time
import threading
from pympler import asizeof

sys.path.append('../src/')
from gledger import GLedger
import utils


# TODO: Move Hard coded variables into yaml 
transaction_line_file = '/Users/stuartwayland/Desktop/stx_lines.pkl'
transaction_agg_file = '/Users/stuartwayland/Desktop/stx_agg.pkl'
raw_outfile = '/Users/stuartwayland/Repos/krv_mapper/data/STX2022-sub100000-Bottom20/raw/stx_agg.pkl'
clean_outfile = '/Users/stuartwayland/Repos/krv_mapper/data/STX2022-sub100000-Bottom20/clean/stx_athena.pkl'
size = -1



def load_model(size):
    lines = pd.read_pickle(transaction_line_file)
    agg = pd.read_pickle(transaction_agg_file)
    if size > 0:
        agg, lines = utils.get_transaction_sample(t_agg=agg, t_lines=lines, num_samples=size)

    return GLedger(t_lines=lines, t_agg=agg)

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
        gledger = load_model(size)

        agg_time = time.time() - start_time
        fit_start = time.time()

        # Run your GLedger's fit method
        gledger.fit()
        # Time to fit 
        fit_time = time.time() - fit_start



        projection_start = time.time()
        projection_dict = gledger.get_projection(sample_perc=20, sample_norm='l2')

        assert projection_dict['clean_data'] is not None 

        # Write projection file 
        with open(clean_outfile, 'wb') as f:
            pickle.dump(projection_dict, f)

        # Write t_agg subset 
        agg_subset = gledger.get_agg_df().iloc[projection_dict['hyperparameters']]
        agg_subset.reset_index(inplace=True, drop=True)
        agg_subset.to_pickle(raw_outfile)

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

    print(f"\n\nRuntime specs for {size} transactions \n")
    print(f"Aggregation time: {agg_time}")
    print(f"Fitting time: {fit_time}")
    print(f"Projection time: {projection_time}")
    print(f"Total Runtime: {end_time}")

