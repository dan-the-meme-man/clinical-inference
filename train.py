import os
import torch
from retrieve_data import *
from NN_NLI import *

def train_batch(model, batch, optimizer, criterion):

        inputs = []
        
        for data_item in batch:
            pass
        
        # zero the gradients
        model.zero_grad()
        
        # forward pass
        output = model(batch)
        
        # compute loss
        loss = criterion(output, batch['label'])
        
        # backward pass
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        return loss.item()

def main():
    
    print('Begin train.py.\n\n\n')
    
    ### TRAINING HYPERPARAMETERS ###
    lr = 1e-3
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
    
    ### LEAVE ALONE? ###
    criterion = torch.nn.BCEWithLogitsLoss()
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-3
    amsgrad = False
    
    # make model
    model = TransformerNLI(specs=specs)
    print(f'Model: {model}\n\n\n')
    
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
    training_data_path = os.path.join('Task-2-SemEval-2024', 'training_data', 'training_data')
    train_json_path = os.path.join(training_data_path, 'train.json')
    dev_json_path = os.path.join(training_data_path, 'dev.json')
    train_dataset = ClinicalDataset(train_json_path)
    dev_dataset = ClinicalDataset(dev_json_path)
    print(f'Loaded {len(train_dataset)} training examples.')
    print(f'{train_dataset.single_entailment} {train_dataset.comparison_entailment} {train_dataset.single_contradiction} {train_dataset.comparison_contradiction}')
    print(f'Loaded {len(dev_dataset)} dev examples.')
    print(f'{dev_dataset.single_entailment} {dev_dataset.comparison_entailment} {dev_dataset.single_contradiction} {dev_dataset.comparison_contradiction}')
    
    ### BE SURE TO SHUFFLE DATA ###
    #shuffle(train_dataset, seed=42)
    #shuffle(dev_dataset, seed=42)
    
    exit()
    
    # epoch loop
    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs}.')
        
        # train
        model.train()
        for i in range(len(train_dataset)):
            
            print(batch)
            exit()
    
if __name__ == '__main__':
    main()