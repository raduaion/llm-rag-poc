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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import sys

index_config = gcfg.newest_index(gcfg.INDEXES_FOLDER)
index = index_config['path']
print(f"Loading FAISS index from {index} [name={index_config['name']}]")
db = FAISS.load_local(index, HuggingFaceEmbeddings(model_name=gcfg.ST_MODEL_NAME), allow_dangerous_deserialization=True)

# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5}
)

query=""
while query != gcfg.EXIT_COMMAND:
    print("-" * 50)
    try:
        query = input("Please enter your question: ")
    except KeyboardInterrupt:
        print(f"\n{gcfg.EXIT_COMMAND}")
        sys.exit(0)
    # docs = db.similarity_search(query)
    docs = retriever.invoke(query)
    print("You asked: " + query)
    print("Found the following information:")
    for doc in docs:
        print("_" * 50)
        print("\t" + doc.page_content + "\n\n")
        print("_" * 20)
        print(doc.metadata)
        print("_" * 50)
    print("-" * 50)
