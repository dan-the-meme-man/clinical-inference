import sys
sys.path.append('.')

import os
import torch
from sklearn.metrics import classification_report

from data.retrieve_data import *

def test(model, item, device):
    
    # tokenize
    indices = [model.sp.EncodeAsIds(item[0])]
    
    # pad
    indices[0] += [model.sp.PieceToId('<pad>')] * (model.max_length - len(indices[0]))

    # convert to tensors and move to device
    indices = torch.tensor(indices)
    indices = indices.to(device)
    
    output = model(indices) # forward pass
        
    del indices # delete to free up memory
    torch.cuda.empty_cache() # empty cache

    logit = torch.sigmoid(output)[0].item()
    class_output = 1 if logit > 0.5 else 0
    #print(logit)

    return class_output

def main():

    # get all models to test
    model_files = os.listdir(os.path.join('nn', 'models'))

    # get dev data for testing
    dev_dataset = get_data(
        'dev',
        use_control=False,
        flatten=True,
        mix=False,
        shuf=False,
        use_indices=False
    )
    l = len(dev_dataset)
    print(f'Loaded {l} dev items.')
    
    # manage device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')

    # test output directory
    test_dir = os.path.join('nn', 'tests')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for model_file in model_files:
        
        # load model, get name, set to eval mode
        model = torch.load(os.path.join('nn', 'models', model_file), map_location=device)
        model.device = device
        model_name = model_file[:-3]
        print(f'Testing {model_name}.')
        model.eval()
        
        preds = []
        labels = []
        
        # test model
        #with torch.no_grad():
        for i in range(l):
            item = dev_dataset[i]
            labels.append(item[1])
            preds.append(test(model, item, device))
                        
        # calculate accuracy
        report = classification_report(labels, preds)

        print(report)

        # write to file
        with open(os.path.join(test_dir, f'{model_name}.txt'), 'w+', encoding='utf-8') as f:
            f.write(report)

if __name__ == '__main__':
    main()
