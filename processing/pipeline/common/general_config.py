#########################################################################
# Copyright 2025 Aion Sigma Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#########################################################################

import os
os.environ['HF_HOME'] = '/opt/hermes/scripts/cache/'

import math
import time
import sys
import datetime
import json

EXIT_COMMAND = "exit"
CLEAR_COMMAND = "clear"

SOURCES_FILE = "sources.txt"
ST_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
LLM_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'
LLM_PADDING_SIDE = 'right'

TS_CHUNK_SIZE = 800
TS_CHUNK_OVERLAP = 0

PIPELINE_QUERY_TEMPERATURE = 0
PIPELINE_QUERY_TEMPERATURE = 0.01
PIPELINE_RESPONSE_TEMPERATURE = 0.2

HF_TOKEN = ""

SIMPLE_PIPELINE_PROMPT_TEMPLATE = """[INST] Instruction: Answer the question based on your knowledge. Here is context to help:

{context}

### question:
{question} [/INST]
"""

CONVERSATIONAL_CONDENSE_QUESTION_PROMPT = """[INST] Given the following Chat History and a Follow Up Question, rephrase the Follow Up Question to be a standalone question, in its original language, following these rules:
- do not rephrase the question if it looks like a greeting or general conversational question
- detect any bias in the question and remove it
- detect binary opositions and remove them
- detect any user preconceptions and remove them
- only provide the rephrased question, no other explanation

### Chat History:
{chat_history}

### Follow Up Question:
{question} [/INST]
"""

CONVERSATIONAL_PROMPT_TEMPLATE = """[INST] You are an assistant answering questions based on the provided Context.
Follow the following rules:
- ignore the Context if the question is not related to the Context
- answer the question with your own words based on the following Context including inline citations with source and page references
- clearly state when you are using the Context and when you are not using the Context

### Context:
{context}

### Question:
{question} [/INST]
"""

CONVERSATIONAL_FINETUNE_QUESTION_PROMPT = """You are an assistant answering questions based on your knowledge on the topic.
Take into account the following chat history when formulating an answer.

### Chat History:
{chat_history}

### Follow Up Question:
{question}
"""

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
INDEXES_FOLDER = "indexes"
LORAS_FOLDER = "loras"

def load_index_info(path, default_name):
    index_config = {'name': default_name}
    index_json = path + ".json"
    if os.path.isfile(index_json):
        f = open(index_json, "r")
        index_config = json.load(f)
        if not 'name' in index_config:
            index_config['name'] = default_name
        f.close()
    index_config["path"] = path
    return index_config

def newest_index(directory):
    all = []
    for d in os.listdir(directory):
        path = os.path.join(directory, d)
        if os.path.isdir(path):
            all.append(path)
    newest = max(all, key=os.path.getmtime)
    path, folder = os.path.split(newest)
    return load_index_info(newest, folder)

def get_indexes(directory):
    all = {}
    for d in os.listdir(directory):
        path = os.path.join(directory, d)
        if not os.path.isdir(path):
            continue
        index_config = load_index_info(path, d)
        all[index_config['name']] = index_config

    return all

def index_file_name(model_name):
    return "faiss_index_" + "".join(c for c in model_name if (c.isalnum() or c in "._-")) + f"_{math.floor(time.time())}"

environment = "dev"
if len(sys.argv) > 1:
    environment = sys.argv[1]

logname = f"llmlog_{environment}_{math.floor(time.time())}.log"
logfile = None

def llm_logfilename(session = None):
    LOGS_DIR = "logs"
    if session is None:
        global logname
        return LOGS_DIR + "/" + logname
    else:
        if not 'logname' in session:
            session['logname'] = LOGS_DIR + "/" + f"llmlog_{environment}_{session['username']}_{math.floor(time.time())}.log"
        return session['logname']

def llm_log(data, session = None):
    global logfile
    global TIME_FORMAT
    LOGS_DIR = "logs"
    if session == None:
        if logfile == None:
            os.makedirs(LOGS_DIR, exist_ok=True)
            logfile = open(llm_logfilename(), "a")
        log_to_use = logfile
    else:
        if not 'logfile' in session:
            os.makedirs(LOGS_DIR, exist_ok=True)
            session['logfile'] = open(llm_logfilename(session), "a")
        log_to_use = session['logfile']

    if session == None:
        print(data)
    log_to_use.write("[" + datetime.datetime.now().strftime(TIME_FORMAT) + "] ")
    log_to_use.write(data)
    log_to_use.write("\n")
    log_to_use.flush()

def start(session):
    session['old_stdout'] = sys.stdout
    # session['old_stderr'] = sys.stderr
    sys.stdout = session['logfile']
    # sys.stderr = session['logfile']

def stop(session):
    sys.stdout = session['old_stdout']
    # sys.stderr = session['old_stderr']
    del session['old_stdout']
    # del session['old_stderr']
