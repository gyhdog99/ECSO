#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path models/llava-1.5-7b \
        --question-file data/llava_eval/MME/llava_mme.jsonl \
        --image-folder data/MME_Benchmark_release_version \
        --answers-file  data/mme/llava-1.5-7b/q/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1
done

wait

output_file=data/mme/llava-1.5-7b/q/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/mme/llava-1.5-7b/q/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    rm data/mme/llava-1.5-7b/q/${CHUNKS}_${IDX}.jsonl 
done


conda activate ~/work/conda_env/llava

cd data/llava_eval/MME

python convert_answer_to_mme.py

cd eval_tool

python calculation.py --results_dir data/mme/llava-1.5-7b/qscap_txt_and_direct_ans/to_eval
