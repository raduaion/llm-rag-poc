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
import base64
import torch
import os

from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import format_document
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.messages import get_buffer_string
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.globals import set_debug, set_verbose

from operator import itemgetter

IDX_KEY = 'indexes'
SECONDARY_IDX_KEY = 'secondary_idx'

from multiprocessing import Lock

class ConversationalPipeline:
    tokenizer = None
    model = None

    mistral_tokenizer = None
    mistral_model = None

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(gcfg.CONVERSATIONAL_CONDENSE_QUESTION_PROMPT)
    FINETUNE_QUESTION_PROMPT = PromptTemplate.from_template(gcfg.CONVERSATIONAL_FINETUNE_QUESTION_PROMPT)

    ANSWER_PROMPT = ChatPromptTemplate.from_template(gcfg.CONVERSATIONAL_PROMPT_TEMPLATE)
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content} {source}")
    sessions = {}
    unique_session = None

    lock = Lock()

    def remove_question(self, param):
        END_TOKEN = "[/INST]\n"
        # gcfg.llm_log(f"{self}: {param}")
        if END_TOKEN in param:
           return param.split(END_TOKEN)[-1]
        else:
            return param

    def print_question(self, param):
        gcfg.llm_log(f"QUESTION: {self} - {param}")
        return param

    def __init__(self, indexes, tag):
        self.tokenizer = ConversationalPipeline.tokenizer
        self.model = ConversationalPipeline.model

        standalone_query_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=gcfg.PIPELINE_QUERY_TEMPERATURE,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        # get rid of additional output: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
        standalone_query_generation_pipeline.tokenizer.pad_token_id = standalone_query_generation_pipeline.model.config.eos_token_id
        standalone_query_generation_llm = HuggingFacePipeline(pipeline=standalone_query_generation_pipeline)

        response_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=gcfg.PIPELINE_RESPONSE_TEMPERATURE,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=2000,
        )
        # get rid of additional output: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
        response_generation_pipeline.tokenizer.pad_token_id = response_generation_pipeline.model.config.eos_token_id
        response_generation_llm = HuggingFacePipeline(pipeline=response_generation_pipeline)

        self.__init_db(gcfg.ST_MODEL_NAME, indexes)

        # Instantiate ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )

        # First we add a step to load memory
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )

        # Now we calculate the standalone question
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | ConversationalPipeline.CONDENSE_QUESTION_PROMPT
            | standalone_query_generation_llm
            | RunnableLambda(self.remove_question),
        }

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | self.retriever[IDX_KEY],
            "question": lambda x: x["standalone_question"],
        }

        if SECONDARY_IDX_KEY in self.retriever:
            retrieved_documents_secondary = {
                "docs": itemgetter("answer") | self.retriever[SECONDARY_IDX_KEY],
                "question": lambda x: x["answer"],
            }

        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: ConversationalPipeline.__combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ConversationalPipeline.ANSWER_PROMPT | response_generation_llm | RunnableLambda(self.remove_question),
            "question": itemgetter("question"),
            "context": final_inputs["context"]
        }

        # And now we put it all together!
        if SECONDARY_IDX_KEY in self.retriever:
            self.final_chain = loaded_memory | standalone_question | retrieved_documents | answer | retrieved_documents_secondary | answer
        else:
            self.final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    @staticmethod
    def initialize():
        ConversationalPipeline.__enable_debug()
        #################################################################
        # Tokenizer
        #################################################################

        model_name=gcfg.LLM_NAME
        gcfg.llm_log(f"Using model: {model_name}")
        login(token = gcfg.HF_TOKEN)

        ConversationalPipeline.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)
        ConversationalPipeline.tokenizer.pad_token = ConversationalPipeline.tokenizer.eos_token
        ConversationalPipeline.tokenizer.padding_side = gcfg.LLM_PADDING_SIDE
        # ConversationalPipeline.tokenizer.clean_up_tokenization_spaces = True

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
                gcfg.llm_log("=" * 80)
                gcfg.llm_log("Your GPU supports bfloat16: accelerate training with bf16=True")
                gcfg.llm_log("=" * 80)

        #################################################################
        # Load pre-trained config
        #################################################################
        ConversationalPipeline.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # bf16=True
        )

        gcfg.llm_log(ConversationalPipeline.__print_number_of_trainable_model_parameters(ConversationalPipeline.model))

    def __init_db(self, st_model_name, indexes):
        gcfg.llm_log(f"Received '{indexes}'")
        self.retriever = {}
        for key in indexes:
            index_entry = indexes[key]
            if len(index_entry) == 0 or index_entry is []:
                if key == SECONDARY_IDX_KEY:
                    continue
                index_config = [gcfg.newest_index(gcfg.INDEXES_FOLDER)]
                index = [index_config[0]['path']]
            else:
                index_config = [ gcfg.load_index_info(idx, idx) for idx in index_entry ]

            db = None
            for idx in index_config:
                gcfg.llm_log(f"Loading FAISS index '{key}' from {idx['path']} [name={idx['name']}]")
                index = FAISS.load_local(idx['path'], HuggingFaceEmbeddings(model_name=st_model_name), allow_dangerous_deserialization=True)
                if db is None:
                    db = index
                else:
                    db.merge_from(index)

            # Connect query to FAISS index using a retriever
            self.retriever[key] = db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 10}
            )

    @staticmethod
    def call_rag(question, session):
        session['question_number'] += 1
        return session['rag'].__call_rag(question)

    def __call_rag(self, question):
        # Prepare the input for the RAG model
        inputs = {"question": question}

        # Invoke the RAG model to get an answer
        result = self.final_chain.invoke(inputs)

        gcfg.llm_log(f"result={result}")

        # Save the current question and its answer to memory for future context
        self.memory.save_context(inputs, {"answer": result["answer"]})

        # result['answer'] = result['answer'][len(question):]

        if not 'question' in result:
            result['question'] = question

        # Return the result
        return result

    @staticmethod
    def __print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

    @staticmethod
    def __combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        #gcfg.llm_log(f"Document strings: {docs}")
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    @staticmethod
    def __enable_debug():
        set_debug(True)
        set_verbose(True)

    @staticmethod
    def get(indexes = {IDX_KEY :[], SECONDARY_IDX_KEY: []}, request = None, create = True):
        # if request:
        #     print("Request headers dictionary:", request.headers)
        # else:
        #     print("No request object")

        tag = 'indexes'

        with ConversationalPipeline.lock:
            username = ConversationalPipeline.__get_username(request)
            if username is None:
                if ConversationalPipeline.unique_session is None and create == True:
                    ConversationalPipeline.unique_session = { 'rag': ConversationalPipeline(indexes, tag), 'question_number': 0, 'username': None, tag: indexes }
                return ConversationalPipeline.unique_session

            gcfg.llm_log(f"New request for user {username}")
            if username in ConversationalPipeline.sessions:
                gcfg.llm_log(f"Using existing session for user {username}")
            elif  create == True:
                gcfg.llm_log(f"Creating new session for user {username}")
                ConversationalPipeline.sessions[username] = { 'rag': ConversationalPipeline(indexes, tag), 'question_number': 0, 'username': username, tag: indexes }
            else:
                return None

            return ConversationalPipeline.sessions[username]
    
    @staticmethod
    def clear(request = None):
        with ConversationalPipeline.lock:
            username = ConversationalPipeline.__get_username(request)

            if username is None:
                qn = 0
                gcfg.llm_log(f"Clearing session.")
                if not ConversationalPipeline.unique_session is None:
                    qn = ConversationalPipeline.unique_session['question_number']
                    ConversationalPipeline.unique_session = None
                return qn
            elif username in ConversationalPipeline.sessions:
                gcfg.llm_log(f"Clearing session for user {username}")
                session = ConversationalPipeline.sessions[username]
                if 'logfile' in session:
                    session['logfile'].close()
                del ConversationalPipeline.sessions[username]
                return session['question_number']

            return 0

    @staticmethod
    def __get_username(request):
        if request is None or not 'authorization' in request.headers:
            gcfg.llm_log("No authorization header found in headers")
            return None
        else:
            return base64.b64decode(request.headers['authorization'].removeprefix('Basic ')).decode('utf-8').split(':')[0]
