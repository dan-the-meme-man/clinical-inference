import sys
sys.path.append('.')

import os
import json
from random import shuffle

from torch.utils.data import Dataset

# control symbols which are used in several scripts
from vocab.control_symbols import control_symbols

# paths to the training data
jsons_path = os.path.join('Task-2-SemEval-2024', 'training_data')
CT_path = os.path.join('data', 'CT_dict.json')

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
        
    Functions:
        flatten(): Returns a flattened string representation of the context. This should be used
        only for building a vocabulary.txt.
"""
class DataItem():
    def __init__(self, uuid, j, text_array):
        self.uuid = uuid
        self.json = j
        self.context = text_array
        self.is_comparison = True if len(text_array) == 2 else False
        self.statement = j['Statement']
        self.label = 1 if j['Label'] == 'Contradiction' else 0
        
    def flatten(self): # flat representation of text for vocab
        if self.is_comparison:
            text_list = self.context[0] + self.context[1]
        else:
            text_list = self.context[0]
        
        text = ''
        for t in text_list:
            text += t + '\n'
            
        return text

"""A PyTorch Dataset for the clinical trial data.

    Args:
        file_path (str): The path to the JSON file to load.
        flatten (bool): If True, replace all whitespace with a single space.
        shuf (bool): If True, shuffle the item-internal texts.
        mix (bool): If True, mix CTRs together in comparison examples. This is ignored if shuf is False.

    Attributes:
        examples (list): A list of DataItems.
        num_single (int): The number of single examples.
        num_comparison (int): The number of comparison examples.
        
    Functions:
        __len__(): Returns the number of examples.
        __getitem__(idx): Returns the example at index idx.
"""
class ClinicalDataset(Dataset):
    def __init__(self,
                 file_path,
                 use_control    = True,
                 flatten        = True,
                 shuf           = False,
                 mix            = False,
                 use_indices    = False):
        
        # define some control symbols
        if use_control:
            self.cls = control_symbols['cls']
            self.trial_sep = control_symbols['trial_sep'] # ignored if mix is True
            self.sent_sep = control_symbols['sent_sep']
            self.statement_sep = control_symbols['statement_sep']
            self.end = control_symbols['end']
        else:
            self.cls = ''
            self.trial_sep = ''
            self.sent_sep = ''
            self.statement_sep = ''
            self.end = ''
        
        # settings about data representation
        self.flatten = flatten # replace all whitespace with a single space
        self.shuf = shuf # shuffle the item-internal texts
        self.mix = mix # mix CTRs together in comparison examples, ignored if not shuf
        
        if self.mix: self.trial_sep = '' # ignore trial_sep if mixing, doesn't make sense
        
        # partition data, may be useful
        self.single_entailment = []
        self.comparison_entailment = []
        self.single_contradiction = []
        self.comparison_contradiction = []
        
        # fetch training/dev examples
        jsons = json.load(open(file_path, 'r', encoding='utf-8'))
        
        # fetch clinical trial data to pull from
        cts = json.load(open(CT_path, 'r', encoding='utf-8'))
        
        """Retrieves the relevant clinical trial data from the CT_dict.json file.
        
            Args:
                j (dict): The JSON object from train.json or dev.json.
                
            Returns:
                list: A list of the relevant clinical trial data.
                The list contains either one or two lists of strings.
        """
        def pull_from_ct_json(j):
            
            # for example j, retrieve the relevant clinical trial
            ct_data_1 = cts[j['Primary_id']][j['Section_id']]
            
            if use_indices: # only keep the relevant sentences?
                ct_data_1_keep = [ct_data_1[x] for x in j['Primary_evidence_index']]
            else: # keep all
                ct_data_1_keep = ct_data_1
            
            # if there is a comparison, retrieve that as well
            if j['Type'] == 'Comparison':
                
                ct_data_2 = cts[j['Secondary_id']][j['Section_id']]
                
                if use_indices: # only keep the relevant sentences?
                    ct_data_2_keep = [ct_data_2[x] for x in j['Secondary_evidence_index']]
                else: # keep all
                    ct_data_2_keep = ct_data_2
                
                return [ct_data_1_keep, ct_data_2_keep]
            
            return [ct_data_1_keep]
        
        # make data examples
        for uuid in jsons: # keys of this dict are the uuids of the examples
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
        
        # flatten categories to make indexing easier
        self.examples = self.single_contradiction + self.single_entailment
        self.examples += self.comparison_contradiction + self.comparison_entailment


    ### TODO: implement slicing
    def __getitem__(self, idx):
        
        item = self.examples[idx] # get the whole data item

        if item.is_comparison:
            text0 = item.context[0]
            text1 = item.context[1]
            if self.shuf: # shuffle text
                if self.mix: # shuffle the CTRs together
                    text = text0 + text1
                    shuffle(text)
                else: # shuffle separately and then concatenate
                    shuffle(text0)
                    shuffle(text1)
            
            # make string representations of the text
            sep = '\n' + self.sent_sep + '\n'
            s = self.cls + '\n' + sep.join(text0)
            s += '\n' + self.trial_sep + '\n'
            s += sep.join(text1)
            s += '\n' + self.statement_sep + '\n' + item.statement + '\n' + self.end + '\n'
            
        else:
            text0 = item.context[0]
            if self.shuf:
                shuffle(text0)
                
            sep = '\n' + self.sent_sep + '\n'
            s = self.cls + '\n' + sep.join(text0)
            s += '\n' + self.statement_sep + '\n' + item.statement + '\n' + self.end + '\n'
                
        if self.flatten:
            s = s.replace('\n', ' ')
            s = s.replace('\t', ' ')
            while '  ' in s:
                s = s.replace('  ', ' ')
        
        s = s.strip()
        
        return s, item.label, item
    
    def __len__(self):
        return len(self.examples)

"""Returns the desired dataset split.

    Args:
        s (str): 'train', 'dev', or 'test'.
        
    Returns:
        ClinicalDataset: The desired dataset.
        
    Raises:
        ValueError: If s is not 'train', 'dev', or 'test'.

"""
def get_data(
        s='train',
        use_control    = True,
        flatten        = True,
        shuf           = False,
        mix            = False,
        use_indices    = False
    ):
    try:
        return ClinicalDataset(
            os.path.join(jsons_path, s + '.json'),
            use_control    = use_control,
            flatten        = flatten,
            shuf           = shuf,
            mix            = mix,
            use_indices    = use_indices
        )
    except:
        raise ValueError('Must be one of "train", "dev", or "test".')

"""The main script can be run to ensure data wrangling has been done.

Raises:
    FileNotFoundError: Throws this error if the data wrangling has not been done properly.
"""
if __name__ == '__main__':
    if not os.path.exists(jsons_path):
        raise FileNotFoundError('training_data folder not found. Run fetch_task.py first.')
    else:
        print('training_data folder found.')
    if not os.path.exists(CT_path):
        raise FileNotFoundError('CT_dict.json not found. Run serialize_cts.py first.')
    else:
        print('CT_dict.json found. You may safely run train.py.')