import os
import re
import ast
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from gpt_eval_utils import load_or_convert_to_dataframe, chat_compeletion_openai

from fastchat.model.model_adapter import get_conversation_template
from fastchat.llm_judge.common import (
    load_judge_prompts,
    Judge,
    two_score_pattern,
    two_score_pattern_backup,
    one_score_pattern,
    one_score_pattern_backup,
    TIE_DELTA,
)

##################################
# Call API
##################################
all_judgments = []
def call_openai_api(match):
    global all_judgments
    if not isinstance(match, list):
        match = [match]
    
    for i in range(len(match)):
        message, judge = match[i]['message'], match[i]['judge']
        model = judge.model_name

        # call openai api
        if "gpt" in model:
            judgment = chat_compeletion_openai(model, message, temperature=0, max_tokens=2048)
        else:
            raise ValueError(f"Invalid judge model name: {model}")
        match[i]['judgment'] = judgment
    all_judgments.append(match)
    return judgment

##################################
# Single match
# Note: After make match, matches will have <message> & <user_prompt>
##################################
def make_match_single(match):
    question, answer, judge = match['question'], match['answer'], match['judge']
    
    if judge.prompt_template["type"] == "single":
        # build user prompt
        model = judge.model_name

        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question,
            answer=answer,
        )
    
        # convert to openai format
        system_prompt = judge.prompt_template["system_prompt"]
        conv = get_conversation_template(model)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        message = conv.to_openai_api_messages()

        # update match
        match['user_prompt'] = user_prompt
        match['message'] = message
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")
    return [match]

def make_match_single_query(match):
    answer, judge = match['answer'], match['judge']
    
    if judge.prompt_template["type"] == "single":
        # build user prompt
        model = judge.model_name

        user_prompt = judge.prompt_template["prompt_template"].format(
            answer=answer,
        )
    
        # convert to openai format
        system_prompt = judge.prompt_template["system_prompt"]
        conv = get_conversation_template(model)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        message = conv.to_openai_api_messages()

        # update match
        match['user_prompt'] = user_prompt
        match['message'] = message
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")
    return [match]



def make_match_single_critic(match):
    question, answer, analysis, judge = match['question'], match['answer'], match['analysis'], match['judge']
    if judge.prompt_template["type"] == "single":
        # build user prompt
        model = judge.model_name

        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question,
            answer=answer,
            critic=analysis,
        )
    
        # convert to openai format
        system_prompt = judge.prompt_template["system_prompt"]
        conv = get_conversation_template(model)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        message = conv.to_openai_api_messages()

        # update match
        match['user_prompt'] = user_prompt
        match['message'] = message
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")
    return [match]

def play_a_match_single(all_judgments):
    all_results = []
    for each in all_judgments:
        each = each[0]
        question, generator, user_prompt, judgment, judge = each['question'], each['generator'], each['user_prompt'], each['judgment'], each['judge']
        image_id = each["image_id"]
        # parse results
        if judge.prompt_template["output_format"] == "[[rating]]":
            rating = 0
            match = re.search(one_score_pattern, judgment)
            if not match:
                match = re.search(one_score_pattern_backup, judgment)

            if match:
                rating = ast.literal_eval(match.groups()[0])
            else:
                rating = 0
        elif judge.prompt_template["output_format"] == "dict":
            rating = ast.literal_eval(judgment)
        else:
            raise ValueError(
                f"invalid output format: {judge.prompt_template['output_format']}"
            )

        # build results
        result = {
            "image_id": image_id,
            "instruction": question,
            "model": generator,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": rating,
            "tstamp": time.time(),
        }
        all_results.append(result)
    return all_results


def play_a_match_single_query(all_judgments):
    all_results = []
    for each in all_judgments:
        each = each[0]
        question, generator, user_prompt, judgment, judge = each['question'], each['generator'], each['user_prompt'], each['judgment'], each['judge']
        image_id = each["image_id"]
        # parse results
        # if judge.prompt_template["output_format"] == "[[rating]]":
        #     rating = 0
        #     match = re.search(one_score_pattern, judgment)
        #     if not match:
        #         match = re.search(one_score_pattern_backup, judgment)

        #     if match:
        #         rating = ast.literal_eval(match.groups()[0])
        #     else:
        #         rating = 0
        # elif judge.prompt_template["output_format"] == "dict":
        #     rating = ast.literal_eval(judgment)
        # else:
        #     raise ValueError(
        #         f"invalid output format: {judge.prompt_template['output_format']}"
        #     )

        rating = 1 if judgment.lower() == "safe" else 0

        # build results
        result = {
            "image_id": image_id,
            "instruction": question,
            "model": generator,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": rating,
            "tstamp": time.time(),
        }
        all_results.append(result)
    return all_results

def overall_result_single(all_results):
    overall_results = dict(
        average_score=np.mean([each['score'] for each in all_results]),
    )
    return overall_results

def overall_result_single_dict(all_results):
    rsts = {}
    for each in all_results:
        scores = each['score']
        for k,v in scores.items():
            if k not in rsts:
                rsts[k] = v
            else:
                rsts[k] += v
    
    num_rst = len(all_results)
    for k in rsts.keys():
        rsts[k] /= num_rst
    
    return rsts

if __name__ == "__main__":
    ##################################
    # Prepare Arguments
    ##################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs", type=str)
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["llm_score", "llm_binary", "llm_binary_strict", "llm_help_in_harm", "vlsafe_3", "llm_score_critic", "llm_binary_critic", "query_safety", "query_political", "query_safety_legal", "query_safety_financial", "query_safety_health", "query_safety_gov", "relevant"],
        help=(
            "Evaluation mode. "
            "`pairwise` runs pairwise comparision against a refernce. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    
    args.output_path = args.output_path if args.output_path else os.path.dirname(args.model_outputs)
    print("{}".format(args).replace(', ', ',\n'))

    ##################################
    # Prepare Answers
    # Note: 
    #   1) We assume both outputs should at least contain <instruction>, <generator>, <output> entries.
    #   2) The pre-processed df_to_annotate should contain,
    #       a) Pairwise: <instruction>, <generator_1>, <generator_2>, <output_1>, <output_2>
    #       b) Single: <instruction>, <generator>, <output>
    ##################################
    model_outputs = load_or_convert_to_dataframe(args.model_outputs)

    df_to_annotate = model_outputs.copy(deep=True)

    ##################################
    # Prepare judges and matches
    # Note:
    #   1) make_match_func:
    #       a) convert input information to openai format for calling APIs
    #       b) output should be [[matches]]
    #   2) play_a_match_func:
    #       a) parse the returned judgments
    #   3) overall_result_func:
    #       a) calculate overall results (e.g. average rating)
    ##################################
    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)
    judge_file_key = os.path.basename(args.judge_file).split('.')[0]
    base_model = 'gpt4' if 'gpt-4' in args.judge_model else 'gpt3.5'
    if args.mode == "llm_score":
        judge = Judge(args.judge_model, judge_prompts["llm_score"])
        make_match_func = make_match_single
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_score_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_score.jsonl"
        )
    elif args.mode == "llm_binary":
        judge = Judge(args.judge_model, judge_prompts["llm_binary"])
        make_match_func = make_match_single
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_binary_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_binary.jsonl"
        )
    elif args.mode == "llm_binary_strict":
        judge = Judge(args.judge_model, judge_prompts["llm_binary_strict"])
        make_match_func = make_match_single
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_binary_strict_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_binary_strict.jsonl"
        )
    elif args.mode == "llm_help_in_harm":
        judge = Judge(args.judge_model, judge_prompts["llm_help_in_harm"])
        make_match_func = make_match_single
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_help_in_harm_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_help_in_harm.jsonl"
        )
    elif args.mode == "vlsafe_3":
        judge = Judge(args.judge_model, judge_prompts["vlsafe_3"])
        make_match_func = make_match_single
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single_dict
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/vlsafe_3_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/vlsafe_3.jsonl"
        )
    elif args.mode == "llm_score_critic":
        judge = Judge(args.judge_model, judge_prompts["llm_score_critic"])
        make_match_func = make_match_single_critic
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_score_critic_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_score_critic.jsonl"
        )
    elif args.mode == "llm_binary_critic":
        judge = Judge(args.judge_model, judge_prompts["llm_binary_critic"])
        make_match_func = make_match_single_critic
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_binary_critic_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/llm_binary_critic.jsonl"
        )
    elif args.mode in ["query_safety", "query_political", "query_safety_legal", "query_safety_financial", "query_safety_health", "query_safety_gov"]:
        judge = Judge(args.judge_model, judge_prompts[args.mode])
        make_match_func = make_match_single_query
        play_a_match_func = play_a_match_single_query
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/{args.mode}_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/{args.mode}.jsonl"
        )
    elif args.mode == "relevant":
        judge = Judge(args.judge_model, judge_prompts["relevant"])
        make_match_func = make_match_single
        play_a_match_func = play_a_match_single
        overall_result_func = overall_result_single
        output_file = (
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/relevant_overall.jsonl",
            f"{args.output_path}/{judge_file_key}/{base_model}_evaluation/relevant.jsonl"
        )

    # Make matches
    matches = []
    for _, row in df_to_annotate.iterrows():

        if "critic" in args.mode:
            match = dict(
                image_id=row["image_id"] if hasattr(row, "image_id") else row["image"],
                question=row['query'],
                answer=row['text'],
                analysis=row['analyze'],
                generator=row['model_id'] if hasattr(row, "model_id") else "dummy_model",
                judge=judge,
            )        
        else:
            match = dict(
                image_id=row["image_id"] if hasattr(row, "image_id") else row["image"],
                question=row['query'],
                answer=row['text'],
                generator=row['model_id'] if hasattr(row, "model_id") else "dummy_model",
                judge=judge,
            )
        matches.append(make_match_func(match))
        
    ##################################
    # Play matches
    ##################################
    print('\n\nStart crawling\n\n')
    if args.parallel == 1:
        for match in tqdm(matches):
            call_openai_api(match)
    else:
        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(call_openai_api, matches), total=len(matches)
            ):
                pass
    
    ##################################
    # Calculate results
    ##################################
    print('\n\nStart parsing and saving\n\n')
    assert len(all_judgments) == len(df_to_annotate)

    all_results = play_a_match_func(all_judgments)
    overall_results = overall_result_func(all_results)
    print(overall_results)
    os.makedirs(os.path.dirname(output_file[0]), exist_ok=True)
    with open(output_file[0], 'w') as f:
        json.dump(overall_results, f, indent=1)
    with open(output_file[1], 'w') as f:
        json.dump(all_results, f, indent=1)
