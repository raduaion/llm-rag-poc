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
import json
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

import common.general_config as gcfg

class DiscourseForumLoader(BaseLoader):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def load(self) -> List[Document]:
        index_path = os.path.join(self.root_dir, 'index.json')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}. Please ensure the directory contains an index.json file.")
        with open(index_path, 'r', encoding='utf-8') as idxf:
            index = json.load(idxf)
        domain = index.get("domain", "")
        documents = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                print(f"Processing file: {dirpath} {filename}")
                if filename.endswith('.json') and '_post_' in filename:
                    full_path = os.path.join(dirpath, filename)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        post = json.load(f)
                        content = post.get("raw")
                        
                        metadata = {
                            "post_id": post.get("id"),
                            "topic_id": post.get("topic_id"),
                            "username": post.get("username"),
                            "created_at": post.get("created_at"),
                            "url": post.get("post_url"),
                            "title": post.get("topic_slug"),
                            "group": post.get("primary_group_name"),
                            "flair": post.get("flair_name"),
                            "score": post.get("score"),
                            "source": domain + post.get("post_url"),
                            # TODO: maybe add reactions and link_counts
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))
        return documents