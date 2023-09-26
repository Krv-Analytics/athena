# Set up file for ATHENA
# WARNING: THIS FILE NEEDS TO BE RUN FROM THE ROOT OF ATHENA 


import os
import shutil 
from dotenv import dotenv_values


if __name__ == "__main__":

    # Color output 
    COLOR_RESET = "\033[0m"  # Reset text color to default
    COLOR_RED = "\033[31m"   # Red text
    COLOR_YELLOW = "\033[33m" # Yellow text
    COLOR_GREEN = "\033[32m"  # Green text


    root = os.getcwd()

    # Define the .env file name and config file name
    env_file_name = ".env"
    config_path = root + "/config.yaml"

    # Define the environment variable values
    env_values = {
        "root": f"{root}",
        "config_file": config_path,
        "azure_connection_string": "<azure_connection_string_value>",
        "THEMA": "<Path to THEMA repository>"
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
        if os.path.isfile(config_path):
            print(COLOR_YELLOW + f"Warning: config.yaml already exists. It will not be overwritten." + COLOR_RESET)
        else:  
            shutil.copy(root + '/templates/config_TEMPLATE.yaml', root + '/config.yaml')
            print(f"Configuration file  has been successfully generated.")

        print(COLOR_GREEN + "Successfully installed Athena" + COLOR_RESET)
    except Exception as e:
        print(COLOR_RED + "Unable to complete configuration script. Terminated with error:" + COLOR_RESET)
        print(e)

    
