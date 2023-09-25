# Set up file for ATHENA
# WARNING: THIS FILE NEEDS TO BE RUN FROM THE ROOT OF ATHENA 


import os
import yaml 
from dotenv import dotenv_values

# Color output 
COLOR_RESET = "\033[0m"  # Reset text color to default
COLOR_RED = "\033[31m"   # Red text
COLOR_YELLOW = "\033[33m" # Yellow text
COLOR_GREEN = "\033[32m"  # Green text


root = os.getcwd()

# Define the .env file name and config file name
env_file_name = ".env"
config_file_name = "config.yaml"


# Define the environment variable values
env_values = {
    "root": f"{root}",
    "azure_connection_string": "<azure_connection_string_value>",
    "THEMA": "<THEMA_value>"
}

# Define the configuration data
config_data = {
    "# Configuration file for data generation and fitting/Projection of General Ledgers": None,
    "# Data Config file for dataset generation": None,
    "azure_storage_account": "polarbox",
    "azure_container": "<company_name>",
    "data_blob": "agg_YEAR-YEAR",
    "subset": {
        "Fiscal Year": "# e.g. Fiscal Year: 2021 will subset the data_blob to only include transactions from 2021"
                        "# Any column from the transaction line dataframe may be subset by an attribute"
    },
    "# Fitting": None,
    "transaction_lines_file": "<path_to_transaction_lines_file>",
    "transaction_agg_file": "<path_to_transaction_agg_file>",
    "sample_size": "<sample_size>",
    "contamination": "<contamination>"
}
try: 
    # Check if .env file already exists
    if os.path.isfile(env_file_name):
        print(COLOR_YELLOW + f"Warning: {env_file_name} already exists. It will not be overwritten." + COLOR_RESET)
    else:
    # Create or update the .env file with the specified values
        with open(".env", "w") as env_file:
            for key, value in env_values.items():
                env_file.write(f"{key}={value}\n")

        print("Environment variables have been written to .env file.")

    # Check if config.yaml file already exists
    if os.path.isfile(config_file_name):
        print(COLOR_YELLOW + f"Warning: {config_file_name} already exists. It will not be overwritten." + COLOR_RESET)
    else:
        # Define the output file name
        output_file = "config.yaml"

        # Write the configuration data to the YAML file
        with open(output_file, "w") as yaml_file:
            yaml.dump(config_data, yaml_file, default_style='"')

        print(f"Configuration file '{output_file}' has been generated.")

    print(COLOR_GREEN + "Successfully installed Athena" + COLOR_RESET)
except Exception as e:
    print(COLOR_RED + "Unable to complete configuration script. Terminated with error:" + COLOR_RESET)
    print(e)

    
