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

"""Trains the model for one batch.
    
    Attributes:
        model: the model to train
        batch: the batch of examples
        optimizer: the optimizer
        criterion: the loss function
        device: the device to use
        update: whether to update the parameters or not
        
    Returns:
        loss.item(): the loss of the batch
"""
def train_batch(model, batch, optimizer, criterion, device, update):
    
    indices = [] # list of lists of token IDs, each list is one example
    labels = [] # list of labels, each label is one example
    
    # tokenize input, get max length
    for item in batch:
        indices.append(model.sp.EncodeAsIds(item[0]))
        labels.append(item[1])
        temp = torch.tensor(indices[-1])
    
    # pad inputs to max length
    for i in range(len(indices)):
        indices[i] += [model.sp.PieceToId('<pad>')] * (model.max_length - len(indices[i]))
    
    # convert to tensors and move to device
    indices = torch.tensor(indices)
    labels = torch.tensor(labels, dtype=float) # BCEWithLogitsLoss expects float
    indices = indices.to(device)
    labels = labels.to(device)
    
    model.zero_grad() # zero the gradients
    
    output = model(indices) # forward pass
    
    loss = criterion(output.view(-1), labels) # compute loss with flattened logits
    
    if update:
        loss.backward() # backward pass
        optimizer.step() # update parameters
    
    return loss.item() # return loss

def main():
    
    print('Begin train.py.\n')
    
    ### TRAINING HYPERPARAMETERS ###
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 2
    epochs = 1
    
    ### MODEL HYPERPARAMETERS ###
    specs = {
        'd_model': 256,
        'num_layers': 6,
        'nhead': 8,
        'dim_feedforward': 1024,
        'embed_dim': 256,
        'dropout': 0.1,
        'activation': 'relu',
        'max_length': 3000
    }
    
    ### LEAVE ALONE? ###
    criterion = torch.nn.BCEWithLogitsLoss()
    betas = (0.9, 0.999)
    eps = 1e-8
    amsgrad = False
    
    # manage device and create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerNLI(specs=specs, device=device).to(device)
    print(f'Model: {model}\n')
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
    train_dataset = get_data(
        'train',
        use_control=False,
        flatten=False,
        shuf=False,
        mix=False,
        use_indices=False
    )
    dev_dataset = get_data(
        'dev',
        use_control=False,
        flatten=False,
        shuf=False,
        mix=False,
        use_indices=False
    )

    # ensure correct counts: 1700 train, 200 dev
    print(f'Loaded {len(train_dataset)} training examples.')
    print(f'Loaded {len(dev_dataset)} dev examples.')
    
    # initialize shuffle seed
    seed(42)
    
    batches_done = 0
    
    ################################## EPOCH ##################################
    for e in range(epochs):
        
        # shuffle data
        shuffle(train_dataset.examples)
        shuffle(dev_dataset.examples)
        
        num_train_batches = len(train_dataset) // batch_size
        num_dev_batches = len(dev_dataset) // batch_size
        
        losses = []
        
        print(f'Begin epoch {e+1}/{epochs}.\n')
        
        ################################## TRAIN ##################################
        model.train()
        
        # loop over batches
        for i in range(num_train_batches):
            
            # get batch
            batch = train_dataset[i * batch_size : (i+1) * batch_size]
            
            # train the model on the batch, record the loss
            losses.append(
                train_batch(
                    model,
                    batch,
                    optimizer,
                    criterion,
                    device,
                    update=True
                )
            )
            batches_done += 1 # increment batch counter
            
            # print progress
            msg = f'Batch {batches_done}/{num_train_batches} loss: {losses[-1]:.4}.'
            msg += f' Average loss: {sum(losses)/len(losses):.4}.\n'
            print(msg)
            
        # dump rest of data into a batch if there is any
        batch = train_dataset[(i+1) * batch_size - 1 : -1]
        if len(batch) > 0:
            losses.append(
                train_batch(
                    model,
                    batch,
                    optimizer,
                    criterion,
                    device,
                    update=True
                )
            )
            batches_done += 1
            msg = f'Remainder batch loss: {losses[-1]:.4}.'
            msg += f' Average loss: {sum(losses)/len(losses):.4}.\n'
            print(msg)
            
        ################################### DEV ###################################
        with torch.no_grad():
            model.eval()
            
            # loop over batches
            for i in range(num_dev_batches):
                
                # get batch
                batch = train_dataset[i * batch_size : (i+1) * batch_size]
                
                # train the model on the batch, record the loss
                losses.append(
                    train_batch(
                        model,
                        batch,
                        optimizer,
                        criterion,
                        device,
                        update=False
                    )
                )
                batches_done += 1 # increment batch counter
                
                # print progress
                msg = f'Batch {batches_done}/{num_train_batches} loss: {losses[-1]:.4}.'
                msg += f' Average loss: {sum(losses)/len(losses):.4}.\n'
                print(msg)
                
            # dump rest of data into a batch if there is any
            batch = train_dataset[(i+1) * batch_size - 1 : -1]
            if len(batch) > 0:
                losses.append(
                    train_batch(
                        model,
                        batch,
                        optimizer,
                        criterion,
                        device,
                        update=True
                    )
                )
                batches_done += 1
                msg = f'Remainder batch loss: {losses[-1]:.4}.'
                msg += f' Average loss: {sum(losses)/len(losses):.4}.\n'
                print(msg)
    
if __name__ == '__main__':
    main()