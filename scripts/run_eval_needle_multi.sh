#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export llama_tokenizer_path="LargeWorldModel/LWM-Text-1M"
export lwm_text_checkpoint="/content/drive/MyDrive/llama_models/DiffLlama-375M/checkpoint-64434"
# jsonl file containing text for haystack. Each line should be a json
# with a single key "text" containing the text.
export haystack_file="/content/drive/MyDrive/data/pg19_subset.jsonl"
export output_file="/content/LWM/output.json"

python3 -u scripts/eval_needle_multi.py \
    --load_checkpoint="$lwm_text_checkpoint" \
    --tokenizer="$llama_tokenizer_path" \
    --max_tokens_per_batch=5000 \
    --output_file="$output_file" \
    --haystack_file="$haystack_file" \
    --context_lengths_min=1000 \
    --context_lengths_max=8000 \
    --n_context_length_intervals=8 \
    --n_document_depth_intervals=5 \
    --n_needles_total=8 \
    --n_needles_retrieve=1 \
    --n_rounds=3
read
