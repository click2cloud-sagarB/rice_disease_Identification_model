import os
from azure.storage.blob import BlobServiceClient

# Retrieve storage account name and access key from environment variables
account_name = os.environ.get('AZURE_SA_NAME')
access_key = os.environ.get('AZURE_SA_ACCESSKEY')

# Construct the connection string using the account name and access key
connection_string = (
    f"DefaultEndpointsProtocol=https;AccountName={account_name};"
    f"AccountKey={access_key};EndpointSuffix=core.windows.net"
)

# Initialize the BlobServiceClient with the constructed connection string
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Example container name
container_name = 'rice-disease-original-dataset'

# Create the container if it doesn't exist
container_client = blob_service_client.get_container_client(container_name)
try:
    container_client.create_container()
except Exception as e:
    print(f"Container already exists: {e}")

def upload_folder(local_path, container_name):
    # Traverse the folder structure
    for root, dirs, files in os.walk(local_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Blob names should include the folder structure
            blob_name = os.path.relpath(file_path, local_path).replace("\\", "/")

            # Create a BlobClient for the file
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

            # Upload the file
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded {file_name} to {blob_name}")

# Path to your folder to upload
local_folder_path = "originaldata"
upload_folder(local_folder_path, container_name)
