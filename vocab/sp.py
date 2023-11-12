import os
import sentencepiece as spm

from control_symbols import control_symbols

train_texts = ['vocab.txt'] # can use ',' separated list
train_texts = [os.path.join('vocab', t) for t in train_texts]

# for a complete list of options see: https://github.com/google/sentencepiece/blob/master/doc/options.md
# be sure to include ' ' at between parameters

params = ''
params += ' --input=' + ','.join(train_texts) # only required parameters
params += ' --model_prefix=' + os.path.join('vocab', 'spm') # specify an output name

params +=  ' --vocab_size=9200' # default: 8000

params += ' --character_coverage=1.0' # default: 0.9995

params += ' --normalization_rule_name=nfkc_cf'
#params = params + ' --normalization_rule_name=nfkc'

#params = params + ' --model_type=bpe'
params = params + ' --model_type=unigram'

params += ' --control_symbols=' + ','.join(list(control_symbols.values()))

params += ' --shrinking_factor=0.95'

train = True
if train:
    spm.SentencePieceTrainer.Train(params)

# load and test vocab    
sp = spm.SentencePieceProcessor()
sp.Load(os.path.join('vocab', 'spm.model'))

print(sp.__dict__)
print(sp.this)

print()
print(sp.DecodeIds([10, 30, 60, 100, 1000, 2000, 4000])) # just some random tokens
