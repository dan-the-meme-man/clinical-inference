import sys
sys.path.append('.')

import os
import json
from random import shuffle, seed

from torch.utils.data import Dataset

# control symbols which are used in several scripts
from vocab.control_symbols import control_symbols

seed(42) # random seed

# paths to the training data
clin_jsons_path = os.path.join('Task-2-SemEval-2024', 'training_data')
CT_path = os.path.join('data', 'CT_dict.json')

"""A helper function to load a JSONL file into a list of JSON objects.

    Args:
        file_path (str): The path to the JSONL file to load.
    
    Returns:
        list: A list of Python dicts.
"""
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

"""A single item of clinical data in a convenient format.
    
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
class ClinDataItem():
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

"""A single item of SNLI data in a convenient format.

    Attributes:
        pair_id (str): The pair ID from the SNLI dataset.
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        label (int): 0 for entailment, 1 for contradiction.
"""
class NLIDataItem():
    def __init__(self, pair_id, sentence1, sentence2, label):
        self.pair_id = pair_id
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label

"""A PyTorch Dataset for the clinical trial data.

    Args:
        file_path (str): The path to the JSON file to load.
        use_control (bool): If True, use control symbols.
        flatten (bool): If True, replace all whitespace with a single space.
        shuf (bool): If True, shuffle the item-internal texts.
        mix (bool): If True, mix CTRs together in comparison examples. This is ignored if shuf is False.
        use_indices (bool): If True, only keep the relevant sentences from the clinical trial data.

    Attributes:
        examples (list): A list of ClinDataItems.
        single_entailment (list): A list of ClinDataItems which are single entailments.
        comparison_entailment (list): A list of ClinDataItems which are comparison entailments.
        single_contradiction (list): A list of ClinDataItems which are single contradictions.
        comparison_contradiction (list): A list of ClinDataItems which are comparison contradictions.
        
    Functions:
        __len__(): Returns the number of examples.
        __getitem__(idx): Returns the example at index idx.
        process_item(item): Helper function for __getitem__().
"""
class ClinicalDataset(Dataset):
    def __init__(self,
                 file_path,
                 shuffle_items  = True,
                 use_control    = True,
                 flatten        = True,
                 shuf           = False,
                 mix            = False,
                 use_indices    = False):
        
        # shuffle
        self.shuffle_items = shuffle_items
        
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
            self.statement_sep = '<stm>'
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
        clin_jsons = json.load(open(file_path, 'r', encoding='utf-8'))
        
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
        for uuid in clin_jsons: # keys of this dict are the uuids of the examples
            j = clin_jsons[uuid]
            text_array = pull_from_ct_json(j)
            if len(text_array) == 2: # comparisons
                if j['Label'] == 'Entailment':
                    self.comparison_entailment.append(ClinDataItem(uuid, j, text_array))
                else:
                    self.comparison_contradiction.append(ClinDataItem(uuid, j, text_array))
            else: # singles
                if j['Label'] == 'Entailment':
                    self.single_entailment.append(ClinDataItem(uuid, j, text_array))
                else:
                    self.single_contradiction.append(ClinDataItem(uuid, j, text_array))
                    
            #if len(self.single_entailment) == 1: break ### DEBUG
        
        # flatten categories to make indexing easier
        self.examples = self.single_contradiction + self.single_entailment
        self.examples += self.comparison_contradiction + self.comparison_entailment
        
        if self.shuffle_items:
            shuffle(self.examples)

    def process_item(self, item):

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

    def __getitem__(self, idx):
        
        item = self.examples[idx] # get the whole data item(s)
        
        # slice
        if type(item) == list:
            return [self.process_item(x) for x in item]
        else:
            return self.process_item(item)
    
    def __len__(self):
        return len(self.examples)
    
"""A PyTorch Dataset for the SNLI dataset.

    Args:
        use_control (bool): If True, use control symbols.

    Attributes:
        examples (list): A list of SNLIDataItems.
        num_entailment (int): The number of entailment examples.
        num_contradiction (int): The number of contradiction examples.
        
    Functions:
        __len__(): Returns the number of examples.
        __getitem__(idx): Returns the example at index idx.
        process_item(item): Helper function for __getitem__().
"""
class NLIDataset(Dataset):
    def __init__(self, shuffle_items=True, which='snli', use_control=True):
        
        self.shuffle_items = shuffle_items
        
        fns = []
        if which == 'snli':
            for split in ('train', 'dev', 'test'):
                path = os.path.join('data', 'snli_data', 'snli_1.0', 'snli_1.0_' + split + '.jsonl')
                fns.append(path)
        elif which == 'mnli':
            for split in ('train', 'dev_matched', 'dev_mismatched'):
                path = os.path.join('data', 'mnli_data', 'multinli_1.0', 'multinli_1.0_' + split + '.jsonl')
                fns.append(path)
        
        # define some control symbols
        self.use_control = use_control
        if self.use_control:
            self.cls = control_symbols['cls']
            self.statement_sep = control_symbols['statement_sep']
            self.end = control_symbols['end']
        else:
            self.cls = ''
            self.statement_sep = '<stm>'
            self.end = ''
        
        self.examples = []
        self.entailments = []
        self.contradictions = []

        for fn in fns: # for each file containing data
                
            jsons = load_jsonl(fn) # load line by line
            
            for j in jsons: # extract information
                gold_label = j['gold_label']
                pair_id = j['pairID']
                sentence1 = j['sentence1']
                sentence2 = j['sentence2']
                
                # skip if neutral or -
                if gold_label in ('neutral', '-'):
                    continue
                
                # make example
                if gold_label == 'entailment':
                    example = NLIDataItem(pair_id, sentence1, sentence2, 0)
                    self.entailments.append(example)
                else:
                    example = NLIDataItem(pair_id, sentence1, sentence2, 1)
                    self.contradictions.append(example)

                # add to examples
                self.examples.append(example)
        
        if self.shuffle_items:
            shuffle(self.examples)
    
    # helper for __getitem__
    def process_item(self, item):
        if self.statement_sep == '':
            sep = ' '
        else:
            sep = ' ' + self.statement_sep + ' '
        s = self.cls + item.sentence1 + sep + item.sentence2 + self.end
        return s, item.label, item
    
    def __getitem__(self, idx):
        
        item = self.examples[idx] # get the whole data item(s)
        
        # slice
        if type(item) == list:
            return [self.process_item(x) for x in item]
        else:
            return self.process_item(item)
    
    def __len__(self):
        return len(self.examples)
    
    # combinable with other datasets
    def __add__(self, other):
        if isinstance(other, NLIDataset):
            # Perform custom addition and return a new instance
            ret = NLIDataset(which=None, use_control=self.use_control)
            ret.contradictions = self.contradictions + other.contradictions
            ret.entailments = self.entailments + other.entailments
            ret.examples = self.examples + other.examples
            return ret
        else:
            # If the other object is of a different type, handle it accordingly
            raise TypeError("Unsupported operand type. Cannot add CustomObject with {}".format(type(other)))

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
        shuffle_items  = True,
        use_control    = True,
        flatten        = True,
        shuf           = False,
        mix            = False,
        use_indices    = False
    ):
    
    if s == 'mnli':
        return NLIDataset(which='mnli', shuffle_items=shuffle_items, use_control=use_control)
    elif s == 'snli':
        return NLIDataset(which='snli', shuffle_items=shuffle_items, use_control=use_control)
    else:
        try:
            return ClinicalDataset(
                os.path.join(clin_jsons_path, s + '.json'),
                shuffle_items  = shuffle_items,
                use_control    = use_control,
                flatten        = flatten,
                shuf           = shuf,
                mix            = mix,
                use_indices    = use_indices
            )
        except:
            raise ValueError('Must be one of "train", "dev", "test", "snli", or "mnli".')

"""The main script can be run to ensure data wrangling has been done.

Raises:
    FileNotFoundError: Throws this error if the data wrangling has not been done properly.
"""
if __name__ == '__main__':
    if not os.path.exists(clin_jsons_path):
        raise FileNotFoundError('training_data folder not found. Run fetch_task.py first.')
    else:
        print('training_data folder found.')
    if not os.path.exists(CT_path):
        raise FileNotFoundError('CT_dict.json not found. Run serialize_cts.py first.')
    else:
        print('CT_dict.json found.')
    for split in ('train', 'dev_matched', 'dev_mismatched'):
        path = os.path.join('data', 'mnli_data', 'multinli_1.0', 'multinli_1.0_' + split + '.jsonl')
        if not os.path.exists(path):
            raise FileNotFoundError(path + ' not found. Run fetch_open_domain.py first.')
        else:
            print('multinli_1.0_' + split + '.jsonl' + ' found.')
    for split in ('train', 'dev', 'test'):
        path = os.path.join('data', 'snli_data', 'snli_1.0', 'snli_1.0_' + split + '.jsonl')
        if not os.path.exists(path):
            raise FileNotFoundError(path + ' not found. Run fetch_open_domain.py first.')
        else:
            print('snli_1.0_' + split + '.jsonl' + ' found.')
    print('You may safely build vocab.txt.')