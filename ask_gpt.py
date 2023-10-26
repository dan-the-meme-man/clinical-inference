import time
import openai

from retrieve_data import get_data

# Set your OpenAI API key
try:
    openai.api_key = open('key.txt', 'r').read().strip()
except:
    raise ValueError('You must provide a valid OpenAI API key in a file called key.txt in the top level directory.')

# Define the prompt you want to send to the model
def main():
    dev_data = get_data('dev')

    prompts = []
    labels = []

    for i in range(len(dev_data)):

        prompt = "Given the following information: \n\n"

        for j in range(len(dev_data[i].context)):
            texts = [dev_data[i].context[j][k] for k in range(len(dev_data[i].context[j]))]
            for text in texts:
                prompt += text + '\n'
                
        prompt += '\n\nDecide whether the following statement is an entailment or a contradiction: '
        
        prompt += dev_data[i].statement
        
        prompt += '\n\nPlease be sure to use only one of those two words in your response.'
        
        prompts.append(prompt)
        
        if dev_data[i].label == 0:
            labels.append('entailment')
        elif dev_data[i].label == 1:
            labels.append('contradiction')
        else:
            raise ValueError('Label must be 0 or 1.')

    tp = []
    fp = []
    tn = []
    fn = []

    # Call the OpenAI API to generate a response
    for i in range(len(prompts)):
        response = openai.Completion.create(
            engine="text-davinci-002",  # Use the GPT-3.5 model
            prompt=prompt,
            max_tokens=50  # You can adjust this value as needed
        )

        # Extract and print the generated text from the response
        generated_text = response.choices[0].text
        
        if 'entailment' in generated_text.lower():
            if labels[i] == 'entailment':
                tp.append(generated_text)
            else:
                fp.append(generated_text)
        elif 'contradiction' in generated_text.lower():
            if labels[i] == 'contradiction':
                tn.append(generated_text)
            else:
                fn.append(generated_text)
                
        time.sleep(3)
                
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)

    print(f'Precision: {prec}, Recall: {rec}, F1: {f1}')

if __name__ == '__main__':
    main()