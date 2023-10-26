import os
import json
from torch.utils.data import Dataset

"""A single item of data in a convenient format.
    
    Attributes:
        uuid (str): UUID from train.json or dev.json.
        json (dict): The JSON object from train.json or dev.json.
        context (list): The relevant text from the clinical trial(s).
            This is organized as a list of length 1 (2 if comparison)
            whose entry is a list of relevant text from the primary
            clinical trial.
        is_comparison (bool): True if the data item is a comparison.
        statement (str): The statement from the JSON object.
        label (int): 0 for contradiction, 1 for entailment.
"""
class DataItem():
    def __init__(self, uuid, j, text_array):
        self.uuid = uuid
        self.json = j
        self.context = text_array
        self.is_comparison = True if len(text_array) == 2 else False
        self.statement = j['Statement']
        self.label = 1 if j['Label'] == 'Contradiction' else 0

"""A PyTorch Dataset for the clinical trial data.

    Attributes:
        examples (list): A list of DataItems.
        num_single (int): The number of single examples.
        num_comparison (int): The number of comparison examples.
        
    Functions:
        __len__(): Returns the number of examples.
        __getitem__(idx): Returns the example at index idx.
"""
class ClinicalDataset(Dataset):
    def __init__(self, file_path):
        
        self.single_entailment = []
        self.comparison_entailment = []
        self.single_contradiction = []
        self.comparison_contradiction = []
        
        # training/dev examples
        jsons = json.load(open(file_path, 'r', encoding='utf-8'))
        
        # clinical trial data to pull from
        cts = json.load(open('CT_dict.json', 'r', encoding='utf-8'))
        
        # make an array of all relevant text from clinical trials
        def pull_from_ct_json(j):
            
            # for example j, retrieve the relevant clinical trial
            ct_data_1 = cts[j['Primary_id']][j['Section_id']]
            ct_data_1_keep = [ct_data_1[x] for x in j['Primary_evidence_index']]
            
            # if there is a comparison, retrieve that as well
            if j['Type'] == 'Comparison':
                ct_data_2 = cts[j['Secondary_id']][j['Section_id']]
                ct_data_2_keep = [ct_data_2[x] for x in j['Secondary_evidence_index']]
                
                return [ct_data_1_keep, ct_data_2_keep]
            
            return [ct_data_1_keep]
        
        # make data examples
        for uuid in jsons:
            j = jsons[uuid]
            text_array = pull_from_ct_json(j)
            if len(text_array) == 2: # comparisons
                if j['Label'] == 'Entailment':
                    self.comparison_entailment.append(DataItem(uuid, j, text_array))
                else:
                    self.comparison_contradiction.append(DataItem(uuid, j, text_array))
            else: # singles
                if j['Label'] == 'Entailment':
                    self.single_entailment.append(DataItem(uuid, j, text_array))
                else:
                    self.single_contradiction.append(DataItem(uuid, j, text_array))
                    
        self.examples = self.single_contradiction + self.single_entailment + self.comparison_contradiction + self.comparison_entailment
        
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

if __name__ == '__main__':    
    if not os.path.exists('CT_dict.json'):
        raise FileNotFoundError('CT_dict.json not found. Run serialize_cts.py first.')
    else:
        print('CT_dict.json found. You may safely run train.py.')