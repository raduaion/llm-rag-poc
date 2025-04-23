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

import common.general_config as gf
import os
from langchain_community.document_loaders import PDFMinerLoader, UnstructuredExcelLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredMarkdownLoader, Docx2txtLoader, UnstructuredODTLoader, UnstructuredWordDocumentLoader, UnstructuredImageLoader, json_loader

def parse(localfiles, text_splitter):
    texts = []
    documents = {}
    for f in localfiles:
        stripped_filename = f.removeprefix("file://")
        path, extension = os.path.splitext(stripped_filename)
        if extension in documents:
            documents[extension].append(stripped_filename)
        else:
            documents[extension] = [stripped_filename]

    loaders = {
        '.pdf': PDFMinerLoader,
        '.docx': Docx2txtLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.odt': UnstructuredODTLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xlsm': UnstructuredExcelLoader,
        '.md': UnstructuredMarkdownLoader,
        '.txt': TextLoader,
        '.json': json_loader.JSONLoader,
        '.png': UnstructuredImageLoader,
        '.jpg': UnstructuredImageLoader,
    }
    extra_params = {
        '.pdf': { "extract_images": True },
        '.json': { "jq_schema": ".[]", "text_content": False },
    }
    return_docs = []
    for extension in documents:
        if not extension in loaders:
            print("Cannot find loader for extension. Discarding {} files with '{}' extension".format(len(documents[extension]), extension))
            continue

        print("Processing {} files with '{}' extension".format(len(documents[extension]), extension))
        loader = loaders[extension]
        for doc in documents[extension]:
            try:
                if extension in extra_params:
                    return_docs.extend(loader(doc, **extra_params[extension]).load())
                else:
                    return_docs.extend(loader(doc).load())
            except Exception as e:
                print("Processing %s failed with '%s'" % (doc, e))
                print("Trying without parameters...")
                return_docs.extend(loader(doc).load())


        # file = open(stripped_filename, "r", encoding="utf-8")
        # texts.append(file.read())
        # file.close()
        # docs_transformed = text_splitter.create_documents(texts)
    return return_docs
