#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="llava-1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /llava/eval/model_mmbench_ask_unsafe.py \
        --model-path models/$MODEL \
        --question-file data/mmbench/$MODEL/q/merge.jsonl \
        --answers-file data/mmbench/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=data/mmbench/$MODEL/tell_ask/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/mmbench/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm adata/mmbench/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl 
done

mkdir -p data/mmbench/$MODEL/tell_ask/to_eval

python /home/ma-user/work/yunhao/code/LLaVA-main/scripts/convert_mmbench_for_submission.py \
    --annotation-file /cache/data/llava_eval/mmbench/mmbench_dev_20230712.tsv \
    --result_file data/mmbench/$MODEL/tell_ask/merge.jsonl \
    --upload_file data/mmbench/$MODEL/tell_ask/to_eval/merge.xlsx 
