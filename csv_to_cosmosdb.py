import os
import pandas as pd
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from uuid import uuid4
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get CosmosDB configuration from environment variables
    cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
    cosmos_key = os.getenv("COSMOS_KEY")
    cosmos_database = os.getenv("COSMOS_DATABASE", "linkedin_data")
    cosmos_container = os.getenv("COSMOS_CONTAINER", "employees")
    
    # CSV file path
    csv_file = os.path.join("csv_file", "new_output.csv")
    
    logger.info(f"Starting import from {csv_file} to CosmosDB")
    
    try:
        # Initialize CosmosDB client
        client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
        
        # Get or create database
        database = client.create_database_if_not_exists(id=cosmos_database)
        logger.info(f"Connected to database: {cosmos_database}")
        
        # Get or create container with /company as partition key
        container = database.create_container_if_not_exists(
            id=cosmos_container,
            partition_key=PartitionKey(path="/company"),
            offer_throughput=400
        )
        logger.info(f"Connected to container: {cosmos_container}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} records from CSV")
        
        # Clean the data - fill NaN values
        df = df.fillna("")
        
        # Process and upload each record
        successful_uploads = 0
        failed_uploads = 0
        
        for _, row in df.iterrows():
            try:
                # Create document structure
                document = {
                    'id': str(uuid4()),  # Generate unique ID
                    'full_name': row['full_name'],
                    'full_name_url': row['full_name_url'],
                    'role': row['role'],
                    'company': row['company'] if row['company'] else "Unknown",  # Ensure partition key exists
                    'time': int(row['time']) if row['time'] and pd.notna(row['time']) else 0,
                    'activity': row['activity'],
                    'interests': row['interests'],
                    'experience_overview': row['experience_overview'],
                    'experience_details': row['experience_details']
                }
                
                # Upload document to CosmosDB
                container.create_item(body=document)
                successful_uploads += 1
                
                # Log progress for every 10 records
                if successful_uploads % 10 == 0:
                    logger.info(f"Progress: {successful_uploads} records uploaded")
                
            except Exception as e:
                logger.error(f"Error uploading record for {row['full_name']}: {str(e)}")
                failed_uploads += 1
        
        logger.info(f"Import completed: {successful_uploads} records uploaded successfully, {failed_uploads} records failed")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
