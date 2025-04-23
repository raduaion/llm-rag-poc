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

import common.general_config as gcfg
import torch
import sys
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence

def remove_question(param):
    END_TOKEN = "[/INST]\n"
    if END_TOKEN in param:
        param = param.split(END_TOKEN)[-1]

    return param

#################################################################
# Tokenizer
#################################################################
login(token = gcfg.HF_TOKEN)

model_name=gcfg.LLM_NAME

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    # temperature=gcfg.PIPELINE_RESPONSE_TEMPERATURE,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)

llm_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = gcfg.SIMPLE_PIPELINE_PROMPT_TEMPLATE

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

index_config = gcfg.newest_index(gcfg.INDEXES_FOLDER)
index = index_config['path']
print (f"Loading FAISS index from {index} [name={index_config['name']}]")
db = FAISS.load_local(index, HuggingFaceEmbeddings(model_name=gcfg.ST_MODEL_NAME), allow_dangerous_deserialization=True)

# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 6}
)

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm_pipeline | RunnableLambda(remove_question)
)

query=""
while query != gcfg.EXIT_COMMAND:
    print("-------------------------------------------------")
    try:
        query = input("Please enter your question: ")
        if query == gcfg.EXIT_COMMAND:
            continue
    except KeyboardInterrupt:
        print(f"\n{gcfg.EXIT_COMMAND}")
        sys.exit(0)
    result = rag_chain.invoke(query)
    print("You asked: " + query)
    print("Answer: " + result)
    print("-------------------------------------------------")
