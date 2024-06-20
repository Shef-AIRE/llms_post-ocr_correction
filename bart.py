from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import os
import pandas as pd
import yaml


# Load BART config from YAML file
def load_config(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config['bart']


# Main function for fine-tuning BART
def main(args):
    # Load config
    config = load_config(args.config)

    # Select model
    model_name = f'facebook/{args.model}'
    output_dir = os.path.join('model', f'{args.model}-ocr')

    # Set up training data
    train = pd.read_csv(args.data)
    train['text'] = train['OCR Text']
    train['labels'] = train['Ground Truth']
    train = Dataset.from_pandas(train)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = train.map(lambda x: tokenizer(x['text'], text_target=x['labels'], max_length=1024, truncation=True), batched=True)

    # Initialise BART
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Fine-tune BART
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    config['learning_rate'] = float(config['learning_rate'])
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **config,
    )
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    # Parse arguments for model/config/data
    parser = argparse.ArgumentParser(description='Fine-tuning BART')
    parser.add_argument('--model', type=str, choices=['bart-base', 'bart-large'],
                        default='bart-base', help='Specify model: bart-base, bart-large')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--data', type=str, help='Path to training data')
    args = parser.parse_args()

    main(args)
