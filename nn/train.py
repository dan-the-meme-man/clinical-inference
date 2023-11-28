import sys
sys.path.append('.')

from random import shuffle, seed

import torch
import matplotlib.pyplot as plt

# control symbols
from vocab.control_symbols import control_symbols

# pulls dataset splits
from data.retrieve_data import get_data

# model architecture
from torch_nn import *

# FOR DEBUGGING
overfit = True

"""Logs a message to the console and to a file.
        
    Attributes:
        msg (str): the message to log
        param_str (str): the file to log to
"""
def log_msg(msg, param_str):
    print(msg)
    with open(os.path.join('nn', 'logs', f'{param_str}.log'), 'a+', encoding='utf-8') as f:
        f.write(msg + '\n')

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
    
    # tokenize input
    for item in batch:
        indices.append(model.sp.EncodeAsIds(item[0].strip()))
        labels.append(item[1])
    
    # pad inputs to max length
    for i in range(len(indices)):
        indices[i] += [model.sp.PieceToId('<pad>')] * (model.max_length - len(indices[i]))
    
    # convert to tensors and move to device
    indices = torch.tensor(indices)
    labels = torch.tensor(labels, dtype=torch.float32) # BCEWithLogitsLoss expects float
    indices = indices.to(device)
    labels = labels.to(device)
    
    model.zero_grad() # zero the gradients
    
    output = model(indices) # forward pass
    
    loss = criterion(output.view(-1), labels) # compute loss with flattened logits
    
    if update:
        loss.backward() # backward pass
        optimizer.step() # update parameters
        
    del indices, labels # delete to free up memory
    torch.cuda.empty_cache() # empty cache
    
    return loss.item() # return loss

def main():
    
    ### TRAINING HYPERPARAMETERS ###
    lr = 7.5e-5
    weight_decay = 7.5e-5
    batch_size = 1
    epochs = 10
    
    ### MODEL HYPERPARAMETERS ###
    specs = {
        'd_model': 256,
        'num_layers': 6,
        'nhead': 8,
        'dim_feedforward': 512,
        'embed_dim': 256,
        'dropout': 0.1,
        'activation': 'relu',
        'max_length': 2400
    }
    
    ### LEAVE ALONE? ###
    criterion = torch.nn.BCEWithLogitsLoss()
    betas = (0.9, 0.999)
    eps = 1e-8
    amsgrad = False
    
    # logging
    logs_dir = os.path.join('nn', 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    # manage device and create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerNLI(specs=specs, device=device).to(device)
    param_str = f'{model.name}_lr_{lr}_wd_{weight_decay}_bs_{batch_size}_ep_{epochs}'
    log_msg(f'Model: {model}\nNumber of params: {model.n_params}\n', param_str)
    log_msg(f'Using device: {device}.\n', param_str)
    
    # make optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad
    )
    log_msg(f'Optimizer: {optimizer}\n', param_str)
   
    # retrieve data
    if not overfit:
        pretrain_dataset = get_data('mnli', use_control=False) + get_data('snli', use_control=False)
        train_dataset = get_data(
            'train',
            use_control=False,
            flatten=True,
            shuf=True,
            mix=False,
            use_indices=False
        )
        dev_dataset = get_data(
            'dev',
            use_control=False,
            flatten=True,
            shuf=False,
            mix=False,
            use_indices=False
        )
    else:
        pretrain_dataset = []
        train_dataset = get_data(
            'train',
            use_control=False,
            flatten=True,
            shuf=True,
            mix=False,
            use_indices=False
        )[:10]
        dev_dataset = train_dataset
        batch_size = 1
        epochs = 100
        print(train_dataset)

    # ensure correct counts: 1700 train, 200 dev
    log_msg(f'Loaded {len(pretrain_dataset)} MNLI/SNLI examples.', param_str)
    log_msg(f'Loaded {len(train_dataset)} clinical training examples.', param_str)
    log_msg(f'Loaded {len(dev_dataset)} clinical dev examples.', param_str)
    
    # initialize shuffle seed
    seed(42)

    # for plotting
    train_losses = []
    dev_losses = []
    
    ################################## EPOCH ##################################
    for e in range(epochs):
        
        # shuffle data
        #shuffle(pretrain_dataset.examples)
        #shuffle(train_dataset.examples)
        #shuffle(dev_dataset.examples)
        
        num_pretrain_batches = len(pretrain_dataset) // batch_size
        num_train_batches = len(train_dataset) // batch_size
        num_dev_batches = len(dev_dataset) // batch_size
        
        log_msg(f'Begin epoch {e+1}/{epochs}.\n', param_str)
        model.train()
        
        ################################ PRETRAIN #################################
        
        if not overfit:
            # loop over batches
            for i in range(num_pretrain_batches):
                
                # get batch
                batch = pretrain_dataset[i * batch_size : (i+1) * batch_size]
                
                # train the model on the batch, record the loss
                train_losses.append(
                    train_batch(
                        model,
                        batch,
                        optimizer,
                        criterion,
                        device,
                        update=True
                    )
                )
                
                # print progress
                msg = f'Batch {(i + 1):6}/{num_pretrain_batches} loss: {train_losses[-1]:8.4f}.'
                msg += f' Average loss: {sum(train_losses)/len(train_losses):8.4f}.\n'
                log_msg(msg, param_str)
                print(len(train_losses))
                
            # dump rest of data into a batch if there is any
            batch = pretrain_dataset[(i+1) * batch_size - 1 : -1]
            if len(batch) > 0:
                train_losses.append(
                    train_batch(
                        model,
                        batch,
                        optimizer,
                        criterion,
                        device,
                        update=True
                    )
                )
                msg = f'Remainder batch loss: {train_losses[-1]:8.4f}.'
                msg += f' Average loss: {sum(train_losses)/len(train_losses):8.4f}.\n'
                log_msg(msg, param_str)
        
        ################################## TRAIN ##################################
        
        # loop over batches
        for i in range(num_train_batches):
            
            # get batch
            batch = train_dataset[i * batch_size : (i+1) * batch_size]
            
            # train the model on the batch, record the loss
            train_losses.append(
                train_batch(
                    model,
                    batch,
                    optimizer,
                    criterion,
                    device,
                    update=True
                )
            )
            
            # print progress
            msg = f'Batch {(i + 1):6}/{num_train_batches} loss: {train_losses[-1]:8.4f}.'
            msg += f' Average loss: {sum(train_losses)/len(train_losses):8.4f}.\n'
            log_msg(msg, param_str)
            
        # dump rest of data into a batch if there is any
        batch = train_dataset[(i+1) * batch_size - 1 : -1]
        if len(batch) > 0:
            train_losses.append(
                train_batch(
                    model,
                    batch,
                    optimizer,
                    criterion,
                    device,
                    update=True
                )
            )
            msg = f'Remainder batch loss: {train_losses[-1]:8.4f}.'
            msg += f' Average loss: {sum(train_losses)/len(train_losses):8.4f}.\n'
            log_msg(msg, param_str)
            
        ################################### DEV ###################################
        model.eval() # TODO: if nan loss persists, try model.train()...
        
        log_msg(f'Begin evaluation {e+1}/{epochs}.\n', param_str)
        
        # loop over batches
        for i in range(num_dev_batches):
            
            # get batch
            batch = dev_dataset[i * batch_size : (i+1) * batch_size]
            
            # train the model on the batch, record the loss
            dev_losses.append(
                train_batch(
                    model,
                    batch,
                    optimizer,
                    criterion,
                    device,
                    update=False
                )
            )
            
            # print progress
            msg = f'Batch {(i + 1):6}/{num_dev_batches} loss: {dev_losses[-1]:8.4f}.'
            msg += f' Average loss: {sum(dev_losses)/len(dev_losses):8.4f}.\n'
            log_msg(msg, param_str)
            
        # dump rest of data into a batch if there is any
        batch = dev_dataset[(i+1) * batch_size - 1 : -1]
        if len(batch) > 0:
            dev_losses.append(
                train_batch(
                    model,
                    batch,
                    optimizer,
                    criterion,
                    device,
                    update=False
                )
            )
            msg = f'Remainder batch loss: {dev_losses[-1]:8.4f}.'
            msg += f' Average loss: {sum(dev_losses)/len(dev_losses):8.4f}.\n'
            log_msg(msg, param_str)
                
    # save model
    models_dir = os.path.join('nn', 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    torch.save(model, os.path.join(models_dir, param_str + '.pt'))
    
    # plot losses
    plots_dir = os.path.join('nn', 'plots')
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(list(range(len(train_losses))), train_losses, label='train', s=2, c='blue')
    plt.subplot(1, 2, 2)
    plt.scatter(list(range(len(dev_losses))), dev_losses, label='dev', s=2, c='orange')
    plt.legend()
    plt.grid()
    plt.title('Losses')
    plt.savefig(os.path.join(plots_dir, param_str + '_losses.png'))
    
if __name__ == '__main__':
    main()
