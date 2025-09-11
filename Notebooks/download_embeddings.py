# scripts/download_glove.py
import requests
import zipfile
import os

def main():
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = "Datasets/Embeddings/glove.6B.zip"
    extract_path = "Datasets/Embeddings/"
    target_file = "glove.6B.100d.txt"
    
    os.makedirs(extract_path, exist_ok=True)
    
    # Check if the file already exists
    if os.path.exists(os.path.join(extract_path, target_file)):
        print("GloVe embeddings already downloaded.")
        return
    
    # Download
    print("Downloading GloVe embeddings...")
    response = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract only the 300d file
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only the 300d file to save space
        with zip_ref.open(target_file) as source, open(os.path.join(extract_path, target_file), 'wb') as target:
            target.write(source.read())
    
    # Remove the zip file to save space
    os.remove(zip_path)
    print("Done! File extracted to:", os.path.join(extract_path, target_file))

if __name__ == "__main__":
    main()