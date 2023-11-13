import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from sklearn.metrics import classification_report
from tqdm import tqdm
'''
You can experiment with:
different gpt models: EleutherAI/gpt-j-6B
different prompts: please change the lines related to prompt in the generate_prompt function
few_shot learning parameters
'''
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def generate_prompt(training_data, dev_data, few_shot=2):
  prompt_length =[]
  prompts =[]
  labels =[]
  with open(training_data) as train, open(dev_data) as dev:
    train = json.load(train)
    dev = json.load(dev)
  train_single = [x for x in train if x['secondary_text']==None]
  train_comp = [x for x in train if x['secondary_text']]
  for i, dt in enumerate(dev):
    dt_primary = ' '.join([x.strip() for x in dt['primary_text']])
    dt_label = dt['label'].strip()
    dt_statement = dt['statement'].strip()
    prompt_text =''
    if dt['secondary_text'] ==None:
      example_data = train_single
    else:
      example_data = train_comp
    prompt_examples = random.sample(example_data, few_shot)
    for p in prompt_examples:
      primary_text = ' '.join([x.strip() for x in p['primary_text']])

      statement = p['statement'].strip()
      label = p['label'].strip()

      if p['secondary_text']:
        secondary_text = ' '.join([x.strip() for x in p['secondary_text']])
        prompt = f'Text1:{primary_text}\tText2:{secondary_text}\tStatement:{statement}\tLabel:{label}'
      else:
        prompt = f'Text:{primary_text}\tStatement:{statement}\tLabel:{label}'

      prompt_text+=prompt
    if dt['secondary_text']:
      dt_secondary = ' '.join([x.strip() for x in dt['secondary_text']])
      test_prompt = f'Text1:{dt_primary}\tText2:{dt_secondary}\tStatement:{dt_statement}\tLabel:'
    else:
      test_prompt = f'Text:{dt_primary}\tStatement:{dt_statement}\tLabel:'

    prompt+=test_prompt
    prompts.append(prompt)
    labels.append(dt_label)

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_length.append(len(input_ids[0]))
    if len(input_ids[0])>1024:
        print(prompt_length)
        print(i)

  return prompts, labels

def pred(input_prompt, model, tokenizer):
  preds = []
  for prompt in tqdm(input_prompt):
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids
      
      gen_tokens = model.generate(
          input_ids[:, :1023],
          do_sample=True,
          temperature=0.9,
          max_length=1024,
	max_new_tokens=1,
      )
      gen_text = tokenizer.batch_decode(gen_tokens)[0]
      preds.append(gen_text)
  return preds


if __name__=="__main__":
    training_data = 'training_data/train_data.json'
    dev_data = 'training_data/dev_data.json'
    prompts, labels= generate_prompt(training_data, dev_data)
    predictions = pred(prompts, model, tokenizer)

    #you might need to change the postprocessing approach based on the output of the data
    predictions = [x.split('Label:')[-1] for x in predictions]
    real_predictions =[]
    for x in predictions:
      if x in 'Contradiction':
        real_predictions.append('Contradiction')
      else:
        real_predictions.append('Entailment')

    final_report = classification_report(labels, real_predictions, target_names=['Contradiction', 'Entailment' ])
    with open('report_prompt.txt', 'a') as report:
        report.write(final_report)
