import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded  %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_bow(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['Sentence'].values
        y_train = train_data['Sentiment'].values
        logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        X_train_bow = vectorizer.fit_transform(X_train)
        logger.debug(f"X_train_tfidf shape after fit_transform: {X_train_bow.shape}")

        with open(os.path.join(get_root_directory(), 'bow_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('bow applied with trigrams and data transformed')
        return X_train_bow, y_train
    
    except Exception as e:
        logger.error('Error during bow transformation: %s', e)
        raise

def train_decisiontree(X_train: np.ndarray, y_train: np.ndarray,min_samples_leaf: int, min_samples_split: float):
    """Train a DecisionTree model."""
    try:
        best_model = DecisionTreeClassifier(
            #max_depth=max_depth,
            min_samples_leaf= min_samples_leaf,
            min_samples_split= min_samples_split
        )
        best_model.fit(X_train, y_train)
        logger.debug('DecisionTree training completed')
        return best_model
    except Exception as e:
        logger.error('Error during Decision Tree model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        #max_depth = params['model_building']['max_depth']
        min_samples_leaf = params['model_building']['min_samples_leaf']
        min_samples_split = params['model_building']['min_samples_split']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'src/datasets/interim/train_processed.csv'))

        # Apply TF-IDF feature engineering on training data
        X_train_bow, y_train = apply_bow(train_data, max_features, ngram_range)

        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_decisiontree(X_train_bow, y_train,min_samples_leaf,min_samples_split)

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, 'decisiontree_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
