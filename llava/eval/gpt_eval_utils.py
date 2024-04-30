import glob
import os
import time
import random
import pathlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
import json

import datasets
import pandas as pd

import openai
import fastchat

AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyData = Union[Sequence[dict[str, Any]], pd.DataFrame, datasets.Dataset]

API_MAX_RETRY = 10000
API_RETRY_SLEEP = 30
API_ERROR_OUTPUT = "$ERROR$"
OPENAI_KEYS = [
    ("your key here", ""),
]
openai.api_key = "your key here"


def convert_to_dataframe(data: AnyData) -> pd.DataFrame:
    """Convert input that AlpacaEval accepts into a dataframe."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, datasets.Dataset):
        return data.data.to_pandas()
    elif isinstance(data, list):
        return pd.DataFrame.from_records(data)
    else:
        # try
        return pd.DataFrame(data)

def load_or_convert_to_dataframe(df=Union[AnyPath, AnyData, Callable], **kwargs):
    """Load a dataframe from a path or convert the input to a dataframe if it's not a path."""
    if isinstance(df, Callable):
        df = df(**kwargs)

    if isinstance(df, (str, os.PathLike, pathlib.Path)):
        df = Path(df)

        # check if it's a globbing pattern
        if "*" in str(df):
            df = pd.concat(
                [load_or_convert_to_dataframe(f, **kwargs) for f in glob.glob(str(df))],
            )
        else:
            suffix = df.suffix
            if suffix == ".json":
                df = pd.read_json(df, **kwargs)
            elif suffix == ".csv":
                df = pd.read_csv(df, **kwargs)
                if df.columns[0] == "Unnamed: 0":
                    df.set_index(df.columns[0], inplace=True)
                    df.index.name = None
            elif suffix == ".tsv":
                df = pd.read_table(df, sep="\t", **kwargs)
            elif suffix == ".jsonl":
                with open(df) as f:
                    lines = f.read().splitlines()
                result = [json.loads(jline) for jline in lines]
                df = pd.DataFrame(result)
            else:
                raise ValueError(f"File format {suffix} not supported.")
    else:
        df = convert_to_dataframe(df, **kwargs)

    return df


def chat_compeletion_openai(model, messages, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    if isinstance(messages, fastchat.conversation.Conversation):
        messages = messages.to_openai_api_messages()
    for _ in range(API_MAX_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                # max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.AuthenticationError) as e:
            print(type(e), e)
            # if "gpt-4" in model:
            #     key, org = random.choice([o for o in OPENAI_KEYS if o[0] != openai.api_key])
            #     print(f"Switching OpenAI API key from {openai.api_key} {openai.organization} to {key} {org}.")
            #     openai.api_key = key
            #     # openai.organization = org
            time.sleep(random.randint(1, API_RETRY_SLEEP))
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(random.randint(1, API_RETRY_SLEEP))
        except Exception as e:
            print(type(e), e)
            time.sleep(random.randint(1, API_RETRY_SLEEP))

    return output