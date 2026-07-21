"""Use IndicTrans2 model for finetuning and inferencing."""
import torch
import os
import sys
import time
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransToolkit import IndicProcessor
from transformers import AutoTokenizer
from argparse import ArgumentParser


BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    total_batches = (len(input_sentences) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    print(f"Starting translation: {len(input_sentences)} sentences, {total_batches} batches (batch_size={BATCH_SIZE})", flush=True)

    for batch_idx, i in enumerate(range(0, len(input_sentences), BATCH_SIZE)):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        batch = [
            s if s.startswith(src_lang) else f"{src_lang} {tgt_lang} {s}"
            for s in batch
        ]

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=256, return_tensors='pt'
        ).to(DEVICE)
        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=128,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

        # Progress logging every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            sents_done = min((batch_idx + 1) * BATCH_SIZE, len(input_sentences))
            sents_per_sec = sents_done / elapsed if elapsed > 0 else 0
            eta = (len(input_sentences) - sents_done) / sents_per_sec if sents_per_sec > 0 else 0
            print(f"  Batch {batch_idx+1}/{total_batches} | {sents_done}/{len(input_sentences)} sents | "
                  f"{sents_per_sec:.1f} sents/sec | ETA: {eta/60:.1f} min", flush=True)

    total_time = time.time() - start_time
    print(f"Translation complete: {len(translations)} sentences in {total_time/60:.1f} min", flush=True)
    return translations


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the source file path')
    parser.add_argument('--model', dest='mod', help='Enter the model folder path')
    parser.add_argument('--output', dest='out', help='Enter the target file path')
    parser.add_argument('--base-model', dest='base_model',
                        default=os.environ.get('BASE_MODEL_PATH', 'ai4bharat/indictrans2-indic-en-1B'),
                        help='HF model id or local path for base model (used for tokenizer).')
    parser.add_argument('--local-files-only', dest='local_files_only', action='store_true',
                        help='Load only from local files (offline mode).')
    args = parser.parse_args()
    indic_indic_ckpt_dir = args.base_model

    print(f"Device: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    print(f"Loading IndicProcessor...", flush=True)
    ip = IndicProcessor(inference=True)

    print(f"Loading model from: {args.mod}", flush=True)
    t0 = time.time()
    indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(args.mod, trust_remote_code=True, local_files_only=args.local_files_only)
    indic_indic_model.to(DEVICE)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"Loading tokenizer from: {indic_indic_ckpt_dir}", flush=True)
    indic_indic_tokenizer = AutoTokenizer.from_pretrained(indic_indic_ckpt_dir, trust_remote_code=True, local_files_only=args.local_files_only)
    print(f"Tokenizer loaded.", flush=True)

    hi_sents = read_lines_from_file(args.inp)
    print(f"Input sentences: {len(hi_sents)}", flush=True)

    src_lang, tgt_lang = "hin_Deva", "eng_Latn"
    or_translations = batch_translate(hi_sents, src_lang, tgt_lang, indic_indic_model, indic_indic_tokenizer, ip)
    write_lines_to_file(or_translations, args.out)
    print(f"Output written to: {args.out}", flush=True)

    # flush the models to free the GPU memory
    del indic_indic_tokenizer, indic_indic_model


if __name__ == '__main__':
    main()
