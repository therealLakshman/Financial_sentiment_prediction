import json
import mlflow
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from mlflow.tracking import MlflowClient

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-43-204-28-132.ap-south-1.compute.amazonaws.com:5000/")

# Configure logging
logging.basicConfig(level=logging.INFO)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_model_registration(model_uri: str, model_name: str):
    """Register model with retry logic."""
    return mlflow.register_model(model_uri, model_name)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded from %s', file_path)
        return model_info
    except Exception as e:
        logging.error('Error loading model info: %s', e)
        raise

def verify_model_artifacts(run_id: str, artifact_path: str):
    """Verify that model artifacts exist before registration."""
    client = MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id, artifact_path)
        if not artifacts:
            raise Exception(f"No artifacts found at path '{artifact_path}' for run '{run_id}'")
        logging.info("Found %s artifacts at path '%s'", len(artifacts), artifact_path)
        return True
    except Exception as e:
        logging.error("Error verifying artifacts: %s", e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry using modern approach."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Verify artifacts exist first
        verify_model_artifacts(model_info['run_id'], model_info['model_path'])
        
        # Register the model with retry logic
        model_version = safe_model_registration(model_uri, model_name)
        
        # Use aliases instead of deprecated stages
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=model_version.version
        )
        
        logging.info('Model %s version %s registered with alias "champion"', 
                    model_name, model_version.version)
        return model_version
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "Financial_model"
        model_version = register_model(model_name, model_info)
        
        print(f"Successfully registered model '{model_name}' version {model_version.version}")
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()