#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="llava-1.5-7b"
SPLIT="VLSafe_harmlessness_examine"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/model_vlsafe_tell_ask.py \
        --model-path models/$MODEL \
        --question-file data/vlsafe/$SPLIT/output/$MODEL/image/q/merge.jsonl \
        --image-folder data/coco/images/train2017 \
        --answers-file data/vlsafe/$SPLIT/output/$MODEL/image/tell_ask/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &
done

wait

output_file=data/vlsafe/$SPLIT/output/$MODEL/image/tell_ask/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/vlsafe/$SPLIT/output/$MODEL/image/tell_ask/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm data/vlsafe/$SPLIT/output/$MODEL/image/tell_ask/${CHUNKS}_${IDX}.jsonl 
done
