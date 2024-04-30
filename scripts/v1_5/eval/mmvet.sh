#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path models/llava-1.5-7b \
        --question-file data/llava_eval/mm-vet/llava-mm-vet.jsonl \
        --image-folder data/mm-vet/images \
        --answers-file  data/mm-vet/llava-1.5-7b/q/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=data/mm-vet/llava-1.5-7b/q/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/mm-vet/llava-1.5-7b/q/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm data/mm-vet/llava-1.5-7b/q/${CHUNKS}_${IDX}.jsonl 
done



mkdir -p data/mm-vet/llava-1.5-7b/qscap_txt_and_direct_ans/to_eval

python /home/ma-user/work/yunhao/code/LLaVA-main/scripts/convert_mmvet_for_eval.py \
    --src data/mm-vet/llava-1.5-7b/qscap_txt_and_direct_ans/merge.jsonl \
    --dst data/mm-vet/llava-1.5-7b/qscap_txt_and_direct_ans/to_eval/merge.json

