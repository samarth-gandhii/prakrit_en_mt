"""Finetuning the indictrans2 model on your dataset."""
from argparse import ArgumentParser
import random
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from IndicTransTokenizer import IndicDataCollator
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq


BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = IndicProcessor(inference=False)


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def create_source_target_pairs(lines):
    """Create source and target pairs from lines."""
    source_sents, target_sents = [], []
    for line in lines:
        src, tgt = line.split('\t')[: 2]
        source_sents.append(src)
        target_sents.append(tgt)
    return source_sents, target_sents


def split_lines(lines, test_size, seed):
    rng = random.Random(seed)
    lines = list(lines)
    rng.shuffle(lines)
    split_idx = int(len(lines) * (1 - test_size))
    return lines[:split_idx], lines[split_idx:]


# def preprocess_function(sources, targets, tokenizer, src_lang, tgt_lang):
#     all_elements = []
#     for src_sent, tgt_sent in zip(sources, targets):
#         if not src_sent.startswith(src_lang):
#             src_sent = f"{src_lang} {tgt_lang} {src_sent}"
#         if not tgt_sent.startswith(tgt_lang):
#             tgt_sent = f"{tgt_sent}"
#         model_inputs = tokenizer(src_sent, truncation=True, padding=False, max_length=256)
#         labels = tokenizer(tgt_sent, truncation=True, padding=False, max_length=256)
#         model_inputs["labels"] = labels["input_ids"]
#         all_elements.append(model_inputs)
#     return all_elements

def preprocess_function(sources, targets, tokenizer, model):

    src_tag = "hin_Deva"
    tgt_tag = "eng_Latn"
    
    tagged_sources = [f"{src_tag} {tgt_tag} {s}" for s in sources]
   
    model_inputs = tokenizer( tagged_sources,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = tokenizer(
        text_target=targets,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    label_ids = labels["input_ids"]

    # Replace pad tokens in labels with -100
    processed_labels = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in label_ids
    ]

    decoder_start_token_id = model.config.decoder_start_token_id

    decoder_input_ids = []

    for label in label_ids:

        shifted = [decoder_start_token_id] + label[:-1]

        # keep exact same length
        shifted = shifted[:len(label)]

        decoder_input_ids.append(shifted)

    model_inputs["labels"] = processed_labels
    model_inputs["decoder_input_ids"] = decoder_input_ids

    return [
        {
            "input_ids": inp,
            "attention_mask": attn,
            "labels": lbl,
            "decoder_input_ids": dec
        }
        for inp, attn, lbl, dec in zip(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["labels"],
            model_inputs["decoder_input_ids"]
        )
    ]


def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    """Initialize the model and the tokenizer."""
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser(description='This program is about finetuning a frame identification model.')
    parser.add_argument('--train', dest='tr', help='Enter the training data in TSV format.')
    parser.add_argument('--test', dest='te', help='Enter the test data in TSV format (optional).')
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.1, help='Test split fraction.')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='Random seed for split.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--epoch', dest='ep', help='Enter the number of epochs.', type=int)
    parser.add_argument('--use-lora', dest='use_lora', action='store_true', help='Use LoRA PEFT')
    args = parser.parse_args()
    quantization = None
    indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
    # indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(indic_indic_ckpt_dir, "indic-indic", quantization)
    indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
        indic_indic_ckpt_dir,
        trust_remote_code=True,
    )
    if args.use_lora:
        print("Configuring PEFT / LoRA adapter layers...")
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        indic_indic_model = get_peft_model(indic_indic_model, peft_config)
        indic_indic_model.print_trainable_parameters()
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # create the tokenized dataset
    if not args.tr:
        raise ValueError("Provide --train TSV.")

    train_dataset = read_lines_from_file(args.tr)
    if args.te:
        test_dataset = read_lines_from_file(args.te)
    else:
        train_dataset, test_dataset = split_lines(train_dataset, args.test_size, args.seed)

    train_source_sents, train_target_sents = create_source_target_pairs(train_dataset)
    test_source_sents, test_target_sents = create_source_target_pairs(test_dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        indic_indic_ckpt_dir,
        trust_remote_code=True,
    )
    src_lang, tgt_lang = "hin_Deva", "eng_Latn"
    train_source_sents = processor.preprocess_batch(train_source_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    train_target_sents = processor.preprocess_batch(train_target_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=True)
    test_source_sents = processor.preprocess_batch(test_source_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    test_target_sents = processor.preprocess_batch(test_target_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=True)
    train_tokenized_dataset = preprocess_function(
        train_source_sents,
        train_target_sents,
        tokenizer,
        indic_indic_model
    )
    test_tokenized_dataset = preprocess_function(
        test_source_sents,
        test_target_sents,
        tokenizer,
        indic_indic_model
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.mod,
        evaluation_strategy="no",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_strategy='no',
        save_total_limit=1,
        num_train_epochs=args.ep,
        predict_with_generate=False,
        label_smoothing_factor=0.1
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=indic_indic_model,
        padding="longest",
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    trainer = Seq2SeqTrainer(
        model=indic_indic_model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # if the model is to be trained from the latest checkpoint
    # always put epochs > no_of_epochs when training for the 1st time
    # trainer.train(resume_from_checkpoint=True)
    # to predict and return the class/label with the highest score
    indic_indic_model.save_pretrained(args.mod + '-final')


if __name__ == '__main__':
    main()
