#!/bin/bash
#SBATCH --job-name="train_5455_proj"
#SBATCH --output="logs/%j_%x.o"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=drd92@georgetown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

echo
echo "Loading CUDA 11.8:"

module load cuda/11.8
module list

echo
echo "Hardware CUDA version:"

nvcc --version

echo
echo "Python version:"

python --version

echo
echo "Installing/ensuring packages (output suppressed, remove -q in bash script to show details)."

pip3 install -qr requirements.txt

echo
echo "Running train.py:"

python3 nn/test.py

echo
echo "Closing CUDA module:"
module unload cuda/11.8
module list

echo
echo "Job complete."
echo
