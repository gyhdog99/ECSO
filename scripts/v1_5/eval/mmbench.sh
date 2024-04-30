#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="llava-1.5-7b"
SPLIT="mmbench_dev_20230712"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
        --model-path models/$MODEL \
        --question-file data/llava_eval/mmbench/$SPLIT.tsv \
        --answers-file data/mmbench/$MODEL/q/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=data/mmbench/$MODEL/q/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/mmbench/$MODEL/q/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm data/mmbench/$MODEL/q/${CHUNKS}_${IDX}.jsonl 
done





mkdir -p data/mmbench/$MODEL/q/to_eval

python scripts/convert_mmbench_for_submission.py \
    --annotation-file data/llava_eval/mmbench/mmbench_dev_20230712.tsv \
    --result_file data/mmbench/llava-1.5-7b/q/merge.jsonl \
    --upload_file data/mmbench/llava-1.5-7b/q/to_eval/merge.xlsx 
