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

from langchain_community.document_transformers import Html2TextTransformer
from common.firefox import AsyncFirefoxLoader

import nest_asyncio
nest_asyncio.apply()

def parse(webarticles):
    # Scrapes the pages above
    loader = AsyncFirefoxLoader(webarticles)
    docs = loader.load()
    # Converts HTML to plain text 
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed
