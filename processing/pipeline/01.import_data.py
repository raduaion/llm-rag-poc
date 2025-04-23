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
import common.webparse as wp
import common.fileparse as fp

import json
from datetime import datetime, timezone
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

sources = open(gcfg.SOURCES_FILE, "r")

webarticles = []
local_files = []

ignored = []

# separate sources
for source in sources.read().splitlines():
    if source.startswith(("http://", "https://")):
        webarticles.append(source)
    elif source.startswith("file://"):
        local_files.append(source)
    elif source.startswith("##"):
        # print(f"{source} is a comment")
        continue
    else:
        ignored.append(source)

sources.close()

index_name = gcfg.index_file_name(gcfg.ST_MODEL_NAME)

# Save used configuration as well
index_config = {
    "sources_file": gcfg.SOURCES_FILE,
    "sources": webarticles + local_files,
    "ignored": ignored,
    "chunk_size": gcfg.TS_CHUNK_SIZE,
    "chunk_overlap": gcfg.TS_CHUNK_OVERLAP,
    "created_on": datetime.now(timezone.utc).strftime(gcfg.TIME_FORMAT),
    "original_name": index_name,
    "model_name": gcfg.ST_MODEL_NAME,
    "name": "Index from " + gcfg.SOURCES_FILE + " created on " + datetime.now(timezone.utc).strftime(gcfg.TIME_FORMAT)
}

os.makedirs(gcfg.INDEXES_FOLDER, exist_ok=True)
with open(os.path.join(gcfg.INDEXES_FOLDER, index_name + ".json"), "w", encoding='utf-8') as f:
    json.dump(index_config, f, ensure_ascii=False, indent=4)

docs_transformed = []
if len(webarticles) != 0:
    print("Processing {} web articles".format(len(webarticles)))
    docs_transformed.extend(wp.parse(webarticles))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=gcfg.TS_CHUNK_SIZE,
                                      chunk_overlap=gcfg.TS_CHUNK_OVERLAP)
if len(local_files) != 0:
    print("Processing {} local files".format(len(local_files)))
    docs_transformed.extend(fp.parse(local_files, text_splitter))

chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name=gcfg.ST_MODEL_NAME))

db.save_local(os.path.join(gcfg.INDEXES_FOLDER, index_name))