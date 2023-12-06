import torch
import torch.nn as nn
from tqdm import tqdm
import random
import time
import json
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import re
import torch.nn.functional as F
from transformers import BertModel
from sklearn.metrics import classification_report
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
model =BertModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

MODEL_SAVE_PATH = "bert.pth"
MAX_LEN=512 # don't change
loss_fn = nn.BCEWithLogitsLoss()  # don't change
EPOCH = 20
HIDDEN_SIZE=100
LEARNING_RATE = 5e-5
BS = 16
DROPOUT =0.1
EPSILON = 1e-8


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



def preprocess_data(text_list):
    processed_text =''
    for text in text_list:
        if text.split() and text.split()[0].isupper():
            continue
        pattern = r'[\u0000-\u001f\u007f-\u009f]'
        # Replace the matched characters with an empty string
        cleaned_text = re.sub(pattern, '', text)
        processed_text+=cleaned_text
    return processed_text

def data_preprocess(data):
    inputs = []
    labels = []
    with open(data) as data_file:
        data_f = json.load(data_file)
        data_f = shuffle(data_f)
        for i, f in enumerate(data_f):
            primary_text = preprocess_data(f['primary_text'])
            statement = f['statement']
            if f['secondary_text']:
                text_type = 'comparison'
                secondary_text = preprocess_data(['secondary_text'])
                input = f'Statement: {statement} Text type: {text_type} Primary text: {primary_text} Secondary text: {secondary_text} '
            else:
                text_type = 'single'
                input = f'Statement: {statement} Text type: {text_type} Primary text: {primary_text} '
            #if i>20:
               # break
            label = label_num_pair[f['label']]
            inputs.append(input)
            labels.append(label)
    return inputs, labels


def preprocessing_for_bert(input):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in tqdm(input):
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

        encoded_sent = tokenizer.encode_plus(
            text=preprocess(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

class BertClassifier(nn.Module):
    def __init__(self, model):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, HIDDEN_SIZE, 1

        # Instantiate BERT model
        self.model = model

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def initialize_model(epochs=EPOCH):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(model)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=LEARNING_RATE,    # Default learning rate
                      eps=EPSILON    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.float().unsqueeze(1))
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels.float().unsqueeze(1))
        val_loss.append(loss.item())

        # Get the predictions
        probs = torch.sigmoid(logits).cpu().numpy()

        # Convert probabilities to binary predictions (0 or 1)
        preds = (probs > 0.5).astype(int).flatten()
        # Calculate the accuracy rate
        accuracy = (preds == b_labels.cpu().numpy()).mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy




def bert_predict(model, test_dataloader):

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    all_logits = []
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
            # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # Apply softmax to calculate probabilities
    probs = torch.sigmoid(all_logits).cpu().numpy()
    print(probs)

    # Convert probabilities to binary predictions (0 or 1)
    predictions = (probs > 0.5).astype(int).flatten()  # Threshold is 0.5

    # all_logits = torch.cat(all_logits, dim=0)
    # # Apply softmax to calculate probabilities
    # probs = F.softmax(all_logits, dim=1).cpu().numpy()
    # predictions = np.argmax(probs, axis=1).tolist()
    # predictions = torch.argmax(all_logits, dim=1).flatten()
    return predictions

if __name__=='__main__':
    set_seed(42)
    # prepare data
    label_num_pair = {"Entailment":0, "Contradiction":1}
    train_input, train_labels = data_preprocess('train_data.json')
    dev_input, dev_labels = data_preprocess('dev_data.json')

    train_input_ids, train_attention_masks = preprocessing_for_bert(train_input)
    dev_input_ids, dev_attention_masks = preprocessing_for_bert(dev_input)

    # Convert other data types to torch.Tensor
    train_labels_ID = torch.tensor(train_labels)
    dev_labels_ID = torch.tensor(dev_labels)


    # Create the DataLoader for our training set
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels_ID)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BS)

    # Create the DataLoader for our validation set
    dev_data = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels_ID)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BS)

    # initialize and train the model
    bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCH)
    train(bert_classifier, train_dataloader, dev_dataloader, epochs=EPOCH, evaluation=True)

    # Save the model state and optimizer state
    torch.save({
        'model_state_dict': bert_classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Include other necessary items like epoch number, loss, etc.
    }, MODEL_SAVE_PATH)

    # load and infer
    model_load_path = MODEL_SAVE_PATH
    checkpoint = torch.load(model_load_path, map_location=device)
    bert_classifier.load_state_dict(checkpoint['model_state_dict'])
    bert_classifier.to(device)  # Move model to the right device
    bert_classifier.eval()  # Set the model to evaluation mode for predictionsptimizer.load_state_dict(checkpoint['optimizer_state_dict']t
    pred = bert_predict(bert_classifier, dev_dataloader)
    print(pred)
    print(dev_labels)
    final_report = classification_report(dev_labels, pred, target_names=['Contradiction', 'Entailment'])
    print(final_report)
    with open('report_bert.txt', 'a') as report:
        report.write(final_report)
