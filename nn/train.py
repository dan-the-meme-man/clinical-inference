import sys
sys.path.append('.')

from random import shuffle, seed

import torch
import sentencepiece as spm

# control symbols
from vocab.control_symbols import control_symbols

# pulls dataset splits
from data.retrieve_data import get_data

# model architecture
from torch_nn import *

# manage device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}.\n')

def train_batch(model, batch, optimizer, criterion):
    
    id_list = [] # list of lists of token IDs, each list is one example
    label_list = [] # list of labels, each label is one example
    max_len = 0
    
    # tokenize input, get max length
    for item in batch:
        id_list.append(model.sp.EncodeAsIds(item[0]))
        label_list.append(item[1])
        if len(id_list[-1]) > max_len:
            max_len = len(id_list[-1])
    
    # pad inputs to max length
    for i in range(len(id_list)):
        id_list[i] += model.sp.EncodeAsIds('<pad>') * (max_len - len(id_list[i]))
    
    # convert to tensors and move to device
    id_list = torch.tensor(id_list)
    label_list = torch.tensor(label_list)
    id_list = id_list.to(device)
    label_list = label_list.to(device)
    
    model.zero_grad() # zero the gradients
    
    output = model(batch) # forward pass
    
    loss = criterion(output, batch['label']) # compute loss
    
    loss.backward() # backward pass
    
    optimizer.step() # update parameters
    
    return loss.item() # return loss

def main():
    
    print('Begin train.py.\n')
    
    ### TRAINING HYPERPARAMETERS ###
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 32
    epochs = 10
    
    ### MODEL HYPERPARAMETERS ###
    specs = {
        'd_model': 256,
        'num_layers': 6,
        'nhead': 8,
        'dim_feedforward': 1024,
        'embed_dim': 256,
        'dropout': 0.1,
        'activation': 'relu'
    }
    model = TransformerNLI(specs=specs)
    print(f'Model: {model}\n')
    
    ### LEAVE ALONE? ###
    criterion = torch.nn.BCEWithLogitsLoss()
    betas = (0.9, 0.999)
    eps = 1e-8
    amsgrad = False
    
    # manage device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    print(f'Using device: {device}.\n')
    
    # make optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay,
                                 amsgrad=amsgrad)
    print(f'Optimizer: {optimizer}\n')
   
    # retrieve data
    train_dataset = get_data('train')
    dev_dataset = get_data('dev')

    # ensure correct counts: 1700 train, 200 dev
    print(f'Loaded {len(train_dataset)} training examples.')
    print(f'Loaded {len(dev_dataset)} dev examples.')
    
    # initialize shuffle seed
    seed(42)
    
    # epoch loop
    for e in range(epochs):
        
        # shuffle data
        shuffle(train_dataset.examples)
        shuffle(dev_dataset.examples)
        
        num_batches = len(train_dataset) // batch_size
        
        losses = []
        
        print(f'Begin epoch {e+1}/{epochs}.\n')
        
        # train
        model.train()
        
        # loop over batches
        for i in range(num_batches):
            
            batch = train_dataset[i * batch_size: (i + 1) * batch_size]
            
            print(batch[0])
            print(len(batch[0]))
            print(len(batch))
            exit()
            
            # train the model on the batch, record the loss
            losses.append(train_batch(model, batch, optimizer, criterion))
    
if __name__ == '__main__':
    main()