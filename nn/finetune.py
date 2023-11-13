import sys
sys.path.append('.')

from random import shuffle

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# pulls dataset splits
from data.retrieve_data import get_data

# manage model
model_name = 'allenai/scibert_scivocab_uncased'
config = AutoConfig.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
print(f'Using model: {model_name}.\n\n\n')

# manage device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Using device: {device}.\n\n\n')

def train_batch(model, batch, optimizer, criterion):
    
    token_ids = []
    labels = []
    for item in batch:
        token_ids.append(tokenizer.encode(
            item[0],
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False
        ))
        labels.append(item[1])
    
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
    model = None
    print(f'Model: {model}\n\n\n')
    
    ### LEAVE ALONE? ###
    criterion = torch.nn.BCEWithLogitsLoss()
    betas = (0.9, 0.999)
    eps = 1e-8
    amsgrad = False
    

    
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