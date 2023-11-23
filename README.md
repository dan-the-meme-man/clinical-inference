# Clinical Inference

## Table of Contents

- [Clinical Inference](#clinical-inference)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [How to run](#how-to-run)

## Introduction

Our submission to SemEval 2024's Task 2: Safe Biomedical Natural Language Inference for Clinical Trials.

This task is documented here: <https://sites.google.com/view/nli4ct/semeval-2024?authuser=0>.

## How to run

All code can be run from the root directory of the repository. Before proceeding, ensure you have the relevant libraries installed:
  
```bash
pip install -r requirements.txt
```

If you are a Linux user, you should have the ```unzip``` util installed. If you are a Windows user, you should
have the Windows tool Expand-Archive instead. The Python script ```fetch_task.py``` will clone the repository and unzip the data files for you. You can also retrieve open domain data with ```fetch_open_domain.py```. You should then run ```serialize_cts.py``` to create the serialized data files:

```bash
python data/fetch_task.py
python data/fetch_open_domain.py
python data/serialize_cts.py
```

You can optionally run ```retrieve_data.py``` to ensure the repo pull and serialization have gone as expected:

```bash
python data/retrieve_data.py
```

You can then optionally run the ChatGPT baseline. Ensure you have a valid API key saved to a file
called "key.txt" in the baseline directory. Then run:

```bash
python baseline/ask_gpt.py
```

Next, build the vocabulary as follows. Vocab size may be controlled in sp.py, among other things.
  
```bash
python vocab/make_vocab.py
python vocab/sp.py
```

Finally, you can train our best model from scratch if you wish. Hyperparameters may be adjusted in this script as well.

```bash
python nn/train.py
```
