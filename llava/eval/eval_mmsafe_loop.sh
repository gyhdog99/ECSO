# SPLIT                             MODE
# 01-Illegal_Activitiy              query_safety
# 02-HateSpeech (162)
# 03-Malware_Generation (43)
# 04-Physical_Harm (143)
# 05-EconomicHarm (132)
# 06-Fraud (153)
# 07-Sex (108)
# 08-Political_Lobbying (180)       query_political
# 09-Privacy_Violence (138)         query_safety
# 10-Legal_Opinion (129)            query_safety_legal
# 11-Financial_Advice (166)         query_safety_financial
# 12-Health_Consultation (108)      query_safety_health
# 13-Gov_Decision (148)             query_safety_gov

MODEL="llava-1.5-7b"
RST_DIR="tell_ask"
for MODE in 'SD_TYPO' 'TYPO' 'SD'; do
    for SPLIT in '01-Illegal_Activitiy' '02-HateSpeech' '03-Malware_Generation' '04-Physical_Harm' '05-EconomicHarm' '06-Fraud' '07-Sex' '09-Privacy_Violence'; do
        python llava/eval/gpt4_vlsafe.py \
        --model_outputs mmsafe/outputs/$SPLIT/$MODE/$MODEL/$RST_DIR/merge.jsonl \
        --judge-file llava/eval/table/vlsafe_judge.jsonl \
        --judge-model gpt-4 \
        --mode query_safety \
        --parallel 30 \
        --output_path data/mmsafe/outputs/$SPLIT/$MODE/$MODEL/$RST_DIR
    done
done