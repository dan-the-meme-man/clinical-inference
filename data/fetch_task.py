import os
import subprocess

git_cmd = 'git clone https://github.com/ai-systems/Task-2-SemEval-2024.git'
cmd1 = 'unzip ' + os.path.join('Task-2-SemEval-2024', 'training_data.zip')
cmd1 += ' -d ' + os.path.join('Task-2-SemEval-2024', 'training_data')
cmd2 = 'powershell Expand-Archive -Path "' + os.path.join('Task-2-SemEval-2024', 'training_data.zip')
cmd2 += '" -DestinationPath "' + os.path.join('Task-2-SemEval-2024', 'training_data') + '"'

if not os.path.exists('Task-2-SemEval-2024'):
    try:
        subprocess.call(git_cmd)
    except:
        print(f'The subprocess call failed. If you have git installed, try the following commands:')
        print(git_cmd)
        print('For Linux users:')
        print('\t' + cmd1)
        print('For Windows users:')
        print('\t' + cmd2)

if not os.path.exists(os.path.join('Task-2-SemEval-2024', 'training_data')):

    code = subprocess.call(cmd1, shell=True)
    if code == 1:
        subprocess.call(cmd2, shell=True)
