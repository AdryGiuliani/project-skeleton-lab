#use wget to download datset
import os
import wget
def download_dataset(url, save_path):
    """
    Downloads a dataset from the specified URL and saves it to the given path.

    Args:
        url (str): The URL of the dataset to download.
        save_path (str): The local path where the dataset should be saved.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, url.split('/')[-1])

    print(f"Downloading dataset from {url} to {filename}...")
    wget.download(url, filename)
    print("\nDownload completed.")

def unzip_dataset(zip_path, extract_to):
    """
    Unzips the dataset from the specified zip file to the given directory.

    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where the contents should be extracted.
    """
    import zipfile

    print(f"Unzipping dataset from {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzipping completed.")

if __name__ == '__main__':
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    save_path = "./data"
    download_dataset(url, save_path)
    zip_path = os.path.join(save_path, "tiny-imagenet-200.zip")
    unzip_dataset(zip_path, save_path)

