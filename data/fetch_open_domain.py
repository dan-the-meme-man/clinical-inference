import os
import subprocess
import requests

def download_zip(url, path, dest):
    
    if os.path.exists(path):
        return
    
    print(f'Downloading {url} to {path}. Please allow a few minutes.')
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)
    
    print(f'Unzipping {path} to {dest}.')
    cmd1 = 'unzip ' + path
    cmd1 += ' -d ' + dest
    cmd2 = 'powershell Expand-Archive -Path "' + path
    cmd2 += '" -DestinationPath "' + dest + '"'

    code = subprocess.call(cmd1, shell=True)
    if code == 1:
        subprocess.call(cmd2, shell=True)

mnli_url = 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip'
mnli_path = os.path.join('data', 'multinli_1.0.zip')
mnli_dest = os.path.join('data', 'mnli_data')

snli_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
snli_path = os.path.join('data', 'snli_1.0.zip')
snli_dest = os.path.join('data', 'snli_data')

download_zip(mnli_url, mnli_path, mnli_dest)
download_zip(snli_url, snli_path, snli_dest)