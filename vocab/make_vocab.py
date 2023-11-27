import sys
sys.path.append('.')

import os

# pulls dataset splits
from data.retrieve_data import get_data

# pull train set
print('Retrieving training data.')
train_dataset = get_data('train', use_control=False, flatten=True)
mnli = get_data('mnli', use_control=False)
snli = get_data('snli', use_control=False)
print('Done.')

x = len(train_dataset)
y = len(mnli)
z = len(snli)

# append all training sentences to string
print('Writing to vocab.txt.')
with open(os.path.join('vocab', 'vocab.txt'), 'w+', encoding='utf-8') as f:
    for i in range(len(train_dataset)):
        f.write(train_dataset[i][0] + '\n')
    for j in range(len(mnli)):
        f.write(mnli[j][0] + '\n')
    for k in range(len(snli)):
        f.write(snli[k][0] + '\n')
print('Done.')