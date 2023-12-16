
# Prompt-based learning and Fine-tuning LLMs

This branch contains code for two experiments: prompt-based learning and fine-tuning LLMs.

## Instructions
Firstly, please create a virtual environment with python=3.8 and install all the dependencies.

The code is easy to read and run and all hyperparameters are listed at the beginning of the code, you can change them while doing the parameter-tuning. 

The ```Prompt``` folder contains the code for prompt-related experiments, while the ```transfer``` folder contains the experiments for fine-tuning+transfer learning. 

Note: If you want to use different BERT models for the task, make sure to change the parameter ```D_in```inside the ```Bert_classifier``` class. For example, if you use ```PubMedBert```, ```D_in``` should be set in *768*, while if you use ```Deberta```, you should set it to *1024*.

