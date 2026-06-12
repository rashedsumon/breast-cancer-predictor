import os
import shutil
import kagglehub
import pandas as pd

def load_cancer_data():
    """
    Downloads the Breast Cancer Wisconsin dataset using kagglehub
    and moves it to a local 'data' directory for easy access.
    Returns a pandas DataFrame.
    """
    local_data_dir = "data"
    local_file_path = os.path.join(local_data_dir, "data.csv")
    
    # If data doesn't exist locally, download it
    if not os.path.exists(local_file_path):
        print("Downloading dataset from Kaggle...")
        os.makedirs(local_data_dir, exist_ok=True)
        
        # Download latest version via kagglehub
        downloaded_path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        
        # kagglehub downloads to a global cache directory. 
        # Find the csv and copy it to our project's data folder.
        for file in os.listdir(downloaded_path):
            if file.endswith(".csv"):
                shutil.copy(os.path.join(downloaded_path, file), local_file_path)
                break
                
        print(f"Dataset successfully saved to: {local_file_path}")
    else:
        print("Dataset already exists locally.")
        
    # Read and return the dataframe
    df = pd.read_csv(local_file_path)
    return df

if __name__ == "__main__":
    # Test the loader locally
    df = load_cancer_data()
    print(f"Dataset loaded successfully with shape: {df.shape}")