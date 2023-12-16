import json
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


dataset = load_dataset('json', data_files={'train': 'Data/train.json', 'dev': 'Data/dev.json'})

def concatenate(example):
    premise = "".join(example["primary_text"])
    if example["secondary_text"] != None:
        premise += "".join(example["secondary_text"])
    prompt = " .Based on this, is the following statement an entailment or contradiction? "
    statement = "".join(example["statement"])
    query = premise + prompt + statement
    example["query"] = query
    label2int = {"Contradiction": 0, "Entailment": 1}
    example['labels'] = label2int[example['label']]
    return example


updated_dataset = dataset.map(concatenate)
updated_dataset = updated_dataset.remove_columns(["primary_text", "secondary_text", "statement"])


print(updated_dataset['train'][0])
# We prefix our tasks with "answer the question"

# print(updated_dataset['train']['label'])
# print(updated_dataset['train']['query'])


# Define the preprocessing function

def preprocess(examples):
   # The "inputs" are the tokenized query:
   inputs = [q for q in examples['query']]
   print(len(inputs))
   model_inputs = tokenizer(inputs, max_length=None, truncation=True)
   #print(model_inputs[0])

   # The "labels" are the tokenized outputs:
   # label2int = {"Contradiction": 0, "Entailment": 1}
   # targets = [label2id[label] for label in examples['label']]
   # labels = [label2int[l] for l in examples['label']]
   # print(len(labels))
   # labels = tokenizer(text_target=labels,
   #                    max_length=None,
   #                    truncation=True)

   #model_inputs["labels"] = labels["input_ids"]
   # model_inputs["labels"] = labels
   # print(len(model_inputs))
   return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = updated_dataset.map(preprocess, batched=True)


#print(tokenized_dataset['train'][0])

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

   return result

# Hyperparameters
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 2

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)






trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["dev"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

trainer.train()
