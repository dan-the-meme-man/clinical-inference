import os
from random import shuffle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from vocab.control_symbols import control_symbols
from data.retrieve_data import get_data
from torch_nn import *

# manage device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_batch(model, batch, optimizer, criterion):

        # TODO: tokenize input using spm.SentencePieceProcessor()
        # TODO: pad input to max length
        # TODO: cast ID list to torch tensor
        # TODO: cast label to torch tensor
        # TODO: send to device
        
        model.zero_grad() # zero the gradients
        
        output = model(batch) # forward pass
        
        loss = criterion(output, batch['label']) # compute loss
        
        loss.backward() # backward pass
        
        optimizer.step() # update parameters
        
        return loss.item() # return loss

def main():
    
    print('Begin train.py.\n\n\n')
    
    ### TRAINING HYPERPARAMETERS ###
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 32
    epochs = 10
    
    ### MODEL HYPERPARAMETERS ###
    specs = {
        'vocab_size': 10000,
        'embed_dim': 300,
        'tf_heads': 6,
        'tf_dim': 300,
        'lin_dim': 300,
        'conv_dim': 300,
        'kernel_size': 3,
        'dropout': 0.1,
        'n_layers': 6
    }
    model = TransformerNLI(specs=specs)
    print(f'Model: {model}\n\n\n')
    
    ### LEAVE ALONE? ###
    criterion = torch.nn.BCEWithLogitsLoss()
    betas = (0.9, 0.999)
    eps = 1e-8
    amsgrad = False
    
    # manage device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    print(f'Using device: {device}.\n\n\n')
    
    # make optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay,
                                 amsgrad=amsgrad)
    print(f'Optimizer: {optimizer}\n\n\n')
   
    # retrieve data
    train_dataset = get_data('train')
    dev_dataset = get_data('dev')

    # ensure correct counts: 1700 train, 200 dev
    print(f'Loaded {len(train_dataset)} training examples.')
    print(f'Loaded {len(dev_dataset)} dev examples.')
    
    # epoch loop
    for e in range(epochs):
        
        # shuffle data
        shuffle(train_dataset, seed=42)
        shuffle(dev_dataset, seed=42)
        
        num_batches = len(train_dataset) // batch_size
        
        losses = []
        
        print(f'Begin epoch {e+1}/{epochs}.\n\n\n')
        
        # train
        model.train()
        
        # loop over batches
        for i in range(num_batches):
            
            batch = []
            
            for j in range(batch_size):
                try:
                    batch.append(train_dataset[i * batch_size + j])
                except:
                    break
            
            # train the model on the batch, record the loss
            losses.append(train_batch(model, batch, optimizer, criterion))
    
if __name__ == '__main__':
    main()