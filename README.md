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

If you are a Linux user, you should have the ```unzip``` util installed. If you are a Windows user, you should
have the Windows tool Expand-Archive instead, which is recognized by Powershell. The Python script ```fetch_task.py``` will clone the repository and unzip the data files for you. You should then run ```serialize_cts.py``` to create the serialized data files:

```bash
python fetch_task.py
python serialize_cts.py
```

Before proceeding, ensure you have the relevant libraries installed:
  
  ```bash
  pip install -r requirements.txt
  ```
