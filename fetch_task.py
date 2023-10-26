import os
import subprocess

if not os.path.exists('Task-2-SemEval-2024'):
    subprocess.call('git clone https://github.com/ai-systems/Task-2-SemEval-2024.git')

if not os.path.exists(os.path.join('Task-2-SemEval-2024', 'training_data')):
    
    cmd1 = 'unzip ' + os.path.join('Task-2-SemEval-2024', 'training_data.zip') + ' -d ' + os.path.join('Task-2-SemEval-2024', 'training_data')
    cmd2 = 'powershell Expand-Archive -Path "' + os.path.join('Task-2-SemEval-2024', 'training_data.zip') + '" -DestinationPath "' + os.path.join('Task-2-SemEval-2024', 'training_data') + '"'
    
    code = subprocess.call(cmd1, shell=True)
    if code == 1:
        subprocess.call(cmd2, shell=True)