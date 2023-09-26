import os 
import yaml

def load_config(config_file):
    """
    Loads config file and builds necessary data paths 
    """ 
    with open(config_file, 'r') as f: 
        config_data = yaml.safe_load(f)
   
    config_data['tlines_path'] = "/data/" + config_data["run_name"] + "/lines.pkl"
    config_data['tagg_path'] = "/data/" + config_data["run_name"] + "/agg.pkl"

    return config_data