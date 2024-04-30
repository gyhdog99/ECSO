SPLIT="VLSafe_harmlessness_examine"
MODEL="llava-1.5-7b"        
RST_DIR="tell_ask"


python llava/eval/gpt4_vlsafe.py \
    --model_outputs data/vlsafe/$SPLIT/output/$MODEL/$MODE/$RST_DIR/merge.jsonl \
    --judge-file llava/eval/table/vlsafe_judge.jsonl \
    --judge-model gpt-4 \
    --mode llm_binary_strict \
    --parallel 30 \
    --output_path data/vlsafe/$SPLIT/output/$MODEL/$MODE/$RST_DIR/ \