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
        statement (str): The statement from the JSON object.
        label (int): 0 for contradiction, 1 for entailment.
"""
class DataItem():
    def __init__(self, uuid, j, text_array):
        self.uuid = uuid
        self.json = j
        self.context = text_array
        self.statement = j['Statement']
        self.label = 0 if j['Label'] == 'Contradiction' else 1

"""A PyTorch Dataset for the clinical trial data.

    Attributes:
        examples (list): A list of DataItems.
        
    Functions:
        __len__(): Returns the number of examples.
        __getitem__(idx): Returns the example at index idx.
"""
class ClinicalDataset(Dataset):
    def __init__(self, file_path):
        
        self.examples = []
        
        # training/dev examples
        jsons = json.load(open(file_path, 'r', encoding='utf-8'))
        
        # clinical trial data to pull from
        cts = json.load(open(os.path.join('data', 'CT_dict.json'), 'r', encoding='utf-8'))
        
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
            self.examples.append(DataItem(uuid, j, text_array))
        
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

if __name__ == '__main__':    
    if not os.path.exists(os.path.join('data', 'CT_dict.json')):
        raise FileNotFoundError('CT_dict.json not found. Run serialize_cts.py first.')
    else:
        print('CT_dict.json found. You may safely run train.py.')