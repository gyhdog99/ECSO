#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="llava-1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /home/ma-user/work/yunhao/code/LLaVA-main/llava/eval/model_vqa_ask_unsafe.py \
        --model-path models/$MODEL \
        --question-file data/mm-vet/$MODEL/q/merge.jsonl \
        --image-folder data/mm-vet/images \
        --answers-file data/mm-vet/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=data/mm-vet/$MODEL/tell_ask/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/mm-vet/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm data/mm-vet/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl 
done

mkdir -p data/mm-vet/$MODEL/tell_ask/to_eval

python /home/ma-user/work/yunhao/code/LLaVA-main/scripts/convert_mmvet_for_eval.py \
    --src data/mm-vet/$MODEL/tell_ask/merge.jsonl \
    --dst data/mm-vet/$MODEL/tell_ask/to_eval/merge.json

