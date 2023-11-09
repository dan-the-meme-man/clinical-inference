import sentencepiece as spm

from control_symbols import control_symbols

train_texts = 'vocab.txt' # can use ',' separated list

# for a complete list of options see: https://github.com/google/sentencepiece/blob/master/doc/options.md
# be sure to include ' ' at between parameters

params = ''
params += ' --input=vocab.txt' # only required parameters
params += ' --model_prefix=spm' # specify an output name

params +=  ' --vocab_size=8000' # default: 8000

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
sp.Load('spm.model')

print(sp.__dict__)
print(sp.this)

print()
print(sp.DecodeIds([10, 30, 60, 100, 1000, 2000, 4000])) # just some random tokens
