import requests
import os
import tarfile
import shutil

# Dataset URL: Leipzig Indonesian Mixed Corpus (100k sentences)
DATA_URL = "https://downloads.wortschatz-leipzig.de/corpora/ind_mixed_2013_100K.tar.gz"
FILENAME = "ind_mixed_2013_100K.tar.gz"
EXTRACT_FOLDER = "ind_mixed_2013_100K"

def download_dataset():
    if not os.path.exists(FILENAME):
        print(f"Downloading Indonesian dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True)
        with open(FILENAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Dataset archive already exists.")

def extract_and_prepare():
    print("Extracting dataset...")
    with tarfile.open(FILENAME, "r:gz") as tar:
        tar.extractall()
    
    sentences_file = os.path.join(EXTRACT_FOLDER, f"{EXTRACT_FOLDER}-sentences.txt")
    
    if os.path.exists(sentences_file):
        print(f"Found sentences file: {sentences_file}")
        print("Cleaning and saving to input.txt...")
        with open(sentences_file, 'r', encoding='utf-8') as fin, \
             open('input.txt', 'w', encoding='utf-8') as fout:
            for line in fin:
                parts = line.split('\t', 1)
                if len(parts) > 1:
                    fout.write(parts[1])
                else:
                    fout.write(line)
        print("Successfully prepared input.txt in Bahasa Indonesia!")
    else:
        print("Error: Sentences file not found.")

    if os.path.exists(EXTRACT_FOLDER):
        shutil.rmtree(EXTRACT_FOLDER)
    if os.path.exists(FILENAME):
        os.remove(FILENAME)

if __name__ == "__main__":
    download_dataset()
    extract_and_prepare()
