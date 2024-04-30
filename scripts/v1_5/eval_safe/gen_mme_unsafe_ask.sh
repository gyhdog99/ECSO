#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="llava-1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/model_vqa_ask_unsafe.py \
        --model-path models/$MODEL \
        --question-file data/mme/$MODEL/q/merge.jsonl \
        --image-folder data/MME_Benchmark_release_version \
        --answers-file data/mme/$MODEL/tell_ask_safe/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=data/mme/$MODEL/tell_ask_safe/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/mme/$MODEL/tell_ask_safe/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm data/mme/$MODEL/tell_ask_safe/${CHUNKS}_${IDX}.jsonl 
done


cd /cache/data/llava_eval/MME

python convert_answer_to_mme.py --result-file $output_file

cd eval_tool

python calculation.py --results_dir data/mme/$MODEL/tell_ask_safe/to_eval
