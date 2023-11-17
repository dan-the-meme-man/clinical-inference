import sys
sys.path.append('.')

import os
import torch

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
    print(logit)

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
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        # test model
        #with torch.no_grad():
        for i in range(l):
            item = dev_dataset[i]
            label = item[1]
            output = test(model, item, device)
            
            if output == 0:
                if label == 0:
                    tn += 1
                elif label == 1:
                    fn += 1
            elif output == 1:
                if label == 0:
                    fp += 1
                elif label == 1:
                    tp += 1
            
            print(f'TP: {tp}. TN: {tn}. FP: {fp}. FN: {fn}.')
                        
        # calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        try:
            precision = tp / (tp + fp)
        except:
            precision = torch.nan
        try:
            recall = tp / (tp + fn)
        except:
            precision = torch.nan
        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except:
            f1 = torch.nan

        print(f'Accuracy: {accuracy}. Precision: {precision}. Recall: {recall}. F1: {f1}.')

        # write to file
        with open(os.path.join(test_dir, f'{model_name}.txt'), 'w+', encoding='utf-8') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'F1: {f1}\n')
            f.write(f'TP: {tp}/{l}\n')
            f.write(f'TN: {tn}/{l}\n')
            f.write(f'FP: {fp}/{l}\n')
            f.write(f'FN: {fn}/{l}\n')

if __name__ == '__main__':
    main()
