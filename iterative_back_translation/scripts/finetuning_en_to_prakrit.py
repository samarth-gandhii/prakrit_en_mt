"""Finetuning IndicTrans2 for English → Prakrit direction.

This is a modified copy of the original finetuning script with:
  - Language tags swapped: eng_Latn → hin_Deva
  - Column reading swapped: reads column 1 (english) as source, column 0 (prakrit) as target
  - Default base model changed to indictrans2-en-indic-1B

This allows the same iterationN_parallel.tsv (prakrit\\tenglish) to be used for
BOTH Pr→En and En→Pr training without needing a separate swapped TSV.
"""
from argparse import ArgumentParser
import os
import random
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransToolkit import IndicProcessor
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = IndicProcessor(inference=False)


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def create_source_target_pairs(lines):
    """Create source and target pairs from lines.

    For En→Pr: column 1 (english) is SOURCE, column 0 (prakrit) is TARGET.
    This allows the same prakrit\\tenglish TSV to be used for both directions.
    """
    source_sents, target_sents = [], []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue

        # SWAPPED: prakrit (col 0) = target, english (col 1) = source
        tgt, src = parts[:2]
        src_clean = src.strip().strip('"')
        tgt_clean = tgt.strip().strip('"')

        # Skip header row
        if tgt_clean.lower() == 'prakrit' and src_clean.lower().startswith('english'):
            continue

        source_sents.append(src)
        target_sents.append(tgt)
    return source_sents, target_sents


def split_lines(lines, test_size, seed):
    rng = random.Random(seed)
    lines = list(lines)
    rng.shuffle(lines)
    split_idx = int(len(lines) * (1 - test_size))
    return lines[:split_idx], lines[split_idx:]


def preprocess_function(sources, targets, tokenizer, model):

    src_tag = "eng_Latn"
    tgt_tag = "hin_Deva"

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


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser(description='Finetuning IndicTrans2 for English → Prakrit translation.')
    parser.add_argument('--train', dest='tr', help='Enter the training data in TSV format.')
    parser.add_argument('--test', dest='te', help='Enter the test data in TSV format (optional).')
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.1, help='Test split fraction.')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='Random seed for split.')
    parser.add_argument(
        '--base_model',
        dest='base_model',
        default=os.environ.get('BASE_MODEL_PATH', 'ai4bharat/indictrans2-en-indic-1B'),
        help='HF model id or local path for base model (used for TOKENIZER).',
    )
    parser.add_argument(
        '--pretrained_weights',
        dest='pretrained_weights',
        default=None,
        help='Path to pre-trained model weights to continue fine-tuning from '
             '(e.g., eng_to_prakrit_2-final). '
             'If not provided, model is loaded from --base_model.',
    )

    print(f"1")
    parser.add_argument(
        '--local-files-only',
        dest='local_files_only',
        action='store_true',
        help='Load only local model files (offline mode).',
    )
    print(f"2")

    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--epoch', dest='ep', help='Enter the number of epochs.', type=int)
    parser.add_argument('--use-lora', dest='use_lora', action='store_true', help='Use LoRA PEFT')
    args = parser.parse_args()
    quantization = None
    indic_indic_ckpt_dir = args.base_model

    # Determine where to load model weights from
    model_load_path = args.pretrained_weights if args.pretrained_weights else indic_indic_ckpt_dir
    print(f"3")
    print(f"Loading MODEL from: {model_load_path}")
    print(f"Loading TOKENIZER from: {indic_indic_ckpt_dir}")

    try:
        # Load MODEL from pretrained weights (or base model if not specified)
        indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_load_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        # ALWAYS load tokenizer from base model (pretrained weights don't include tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            indic_indic_ckpt_dir,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load model/tokenizer. If the cluster is offline, pass a local path via "
            "--base_model (for tokenizer) and --pretrained_weights (for model) and enable --local-files-only."
        ) from exc
    if not args.tr:
        raise ValueError("Provide --train TSV.")

    train_dataset = read_lines_from_file(args.tr)
    if args.te:
        test_dataset = read_lines_from_file(args.te)
    else:
        train_dataset, test_dataset = split_lines(train_dataset, args.test_size, args.seed)
    print(f"4")

    train_source_sents, train_target_sents = create_source_target_pairs(train_dataset)
    print(f"5")
    test_source_sents, test_target_sents = create_source_target_pairs(test_dataset)
    print(f"6")
    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    train_source_sents = processor.preprocess_batch(train_source_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    train_target_sents = processor.preprocess_batch(train_target_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=True)
    test_source_sents = processor.preprocess_batch(test_source_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False)
    test_target_sents = processor.preprocess_batch(test_target_sents, src_lang=src_lang, tgt_lang=tgt_lang, is_target=True)
    print(f"7")
    train_tokenized_dataset = preprocess_function(
        train_source_sents,
        train_target_sents,
        tokenizer,
        indic_indic_model
    )
    print(f"8")
    test_tokenized_dataset = preprocess_function(
        test_source_sents,
        test_target_sents,
        tokenizer,
        indic_indic_model
    )
    print(f"9")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.mod,
        eval_strategy="no",
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_strategy='no',
        save_total_limit=1,
        num_train_epochs=args.ep,
        predict_with_generate=False,
        label_smoothing_factor=0.1,
    )
    print(f"10")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=indic_indic_model,
        padding="longest",
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    print(f"11")
    trainer = Seq2SeqTrainer(
        model=indic_indic_model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    print(f"12")
    trainer.train()
    print(f"13")
    indic_indic_model.save_pretrained(args.mod + '-final')
    print(f"14")

if __name__ == '__main__':
    main()
