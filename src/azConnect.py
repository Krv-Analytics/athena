from azure.storage.blob import (
BlobServiceClient,
ContainerClient,
BlobClient
)
import gledger as GLedger



class azConnect: 
    """

    TODO: Move this into docs. Generalize this doc string for any account. 


    A Wrapper class for connecting to azure storage account containers. 


    Members
    ------- 

    storage_account: str
        
        Defaulted to 'polarbox', which is storage account associated with all general ledger
        datasets 

    container_name: str

        The name of the container. Standard is all lower case, no symbols, no spaces company name. 

        
    connection_str: 

        The container connection string. 

    blob_name: 

        DataSets: 
        
        Defaulted to 'tagg_XXXX' or 'tline_XXXX', standing for aggregated transactions or transaction
        lines. Every data container in 'polarbox' is expected to contain at least one 'tagg' and one 
        'tline' blob. 


        Models: 
        Naming convention is 'model_XXXX' 


    Member functions
    ----------------
    create_container(): 
    
    get_blob(): 

    load_blob(): 


    """

    def __init__(self, storage_account:str, container_name:str, connection_string:str ): 
        """
        Must supply an Azure storage account name and connection string. Can be found 
        in Access Keys tab. 

        """

        self.storage_account = storage_account
        self.container_name = container_name
        self._connection_string = connection_string

    
    def create_container(self, container_name=None):
        """
        Creates a new container called self.container_name. If a new name is supplied, 
        self.container_name will be updated. 

        """ 

        if container_name is not None: 
            self.container_name = container_name 
        
        container_client = ContainerClient.from_connection_string(conn_str=self._connection_string, container_name=self.container_name)

        container_client.create_container()

    def get_blob(self, blob_name:str):
        """
        Returns data stored in `blob_name` in self.container_name. 
        """
        # STUB! 
        return 
        

    

    


        




