#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="llava-1.5-7b"

for SPLIT in '01-Illegal_Activitiy' '02-HateSpeech' '03-Malware_Generation' '04-Physical_Harm' '05-EconomicHarm' '06-Fraud' '07-Sex' '08-Political_Lobbying' '09-Privacy_Violence' '10-Legal_Opinion' '11-Financial_Advice' '12-Health_Consultation' '13-Gov_Decision'; do
    for MODE in 'SD_TYPO' 'TYPO' 'SD'; do
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/model_mmsafe_tell_ask.py \
                --model-path models/$MODEL \
                --question-file data/mmsafe/outputs/$SPLIT/$MODE/$MODEL/q/merge.jsonl \
                --image-folder data/mmsafe/images/$SPLIT \
                --answers-file data/mmsafe/outputs/$SPLIT/$MODE/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl \
                --num-chunks $CHUNKS \
                --mode $MODE \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode vicuna_v1 &
        done

        wait

        output_file=data/mmsafe/outputs/$SPLIT/$MODE/$MODEL/tell_ask/merge.jsonl

        # Clear out the output file if it exists.
        > "$output_file"

        # Loop through the indices and concatenate each file.
        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat data/mmsafe/outputs/$SPLIT/$MODE/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl >> "$output_file"
            rm data/mmsafe/outputs/$SPLIT/$MODE/$MODEL/tell_ask/${CHUNKS}_${IDX}.jsonl 
        done
    done
done

