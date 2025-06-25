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

import threading
import queue
import requests
import json
import os
import time
from typing import Callable, Optional, Any
from datetime import datetime

class RateLimiter:
    """
    Simple token-bucket rate limiter.
    Allows up to `rate` requests per second.
    """
    def __init__(self, rate: float):
        self._interval = 1.0 / rate if rate > 0 else 0
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def wait(self):
        """Block until the next request is allowed."""
        with self._lock:
            now = time.monotonic()
            if self._next_allowed > now:
                time.sleep(self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._interval

class DiscourseDownloader:
    class Options:
        def __init__(
            self,
            api_key: str = "",
            verbose: bool = False,
            num_threads: int = 4,
            page_size: int = 30,
            rate_limit: float = 0.0,
            save_path: Optional[str] = None,
            on_post: Optional[Callable[[int, int, Any], None]] = None,
        ):
            """
            :param api_key: e.g. "?api_key=XYZ&api_username=bot"
            :param verbose: Print debug logs.
            :param num_threads: Number of worker threads.
            :param page_size: Topics per page when listing categories.
            :param rate_limit: Max requests per second (0 = unlimited).
            :param save_path: Base directory to save posts to files.
            :param on_post: Callback(topic_id, post_id, data) after each post.
            """
            self.api_key     = api_key
            self.verbose     = verbose
            self.num_threads = num_threads
            self.page_size   = page_size
            self.rate_limit  = rate_limit
            self.save_path   = save_path
            self.on_post     = on_post

    def __init__(self, host: str, opts: Options):
        self.host    = host.rstrip('/')
        self.opts    = opts
        self.session = requests.Session()
        self._task_q = queue.Queue()
        self._output_lock = threading.Lock()
        self._rate_limiter = RateLimiter(opts.rate_limit) if opts.rate_limit > 0 else None
        self.threads_stop = threading.Event()
        self.threads = []

        # Create base save directory if needed
        if self.opts.save_path:
            os.makedirs(self.opts.save_path, exist_ok=True)

    def _start_threads(self):
        self.threads_stop.clear()
        # Start workers
        for _ in range(self.opts.num_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            self.threads.append(t)
            t.start()

    def wait_for_downloads(self):
        self.threads_stop.set()
        for t in self.threads:
            if t.is_alive():
                self._log(f"Waiting for thread {t.name} to finish...")
            t.join()
            self._log(f"Done with thread {t.name}")
        self.threads = []

    def _log(self, *args):
        if self.opts.verbose:
            with self._output_lock:
                print(f"[DEBUG] {threading.current_thread().name}", *args)

    def _build_url(self, path: str) -> str:
        url = f"https://{self.host}{path}"
        if self.opts.api_key:
            url += self.opts.api_key
        return url

    def get_category_topics(self, category_id: int) -> list:
        """
        Fetch all topic IDs in a given category by paging until no more topics.
        """
        topics = []
        page = 0
        while True:
            path = f"/c/{category_id}/l/latest.json?page={page}"
            url  = self._build_url(path)
            self._log("GET", url)
            if self._rate_limiter:
                self._rate_limiter.wait()
            resp = self.session.get(url)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                self._log(f"Error fetching topics for category {category_id}: {e}")
                page += 1
                continue
            data = json.loads(resp.content)
            page_topics = data.get("topic_list", {}).get("topics", [])
            if not page_topics:
                break
            for t in page_topics:
                topics.append(int(t.get("id", 0)))
            page += 1
        self._log(f"Found {len(topics)} topics in category {category_id}")
        return topics
    
    def _worker(self):
        """Continuously fetch posts until queue is empty."""
        while not (self.threads_stop.is_set() and self._task_q.empty()):
            try:
                topic_id, save_dir, data = self._task_q.get(timeout=5)
                # self._log("Working on topic", topic_id)
            except queue.Empty:
                # self._log("Empty queue")
                continue
            if type(data) is int:
                post_id = data
                download = True
            else:
                post_id = data.get('id', None)
                download = False

            try:
                if save_dir:
                    ext = 'json'
                    filepath = os.path.join(save_dir, f"topic_{topic_id}_post_{post_id}.{ext}")
                    if os.path.isfile(filepath):
                        # self._log(f"Skipping cached {filepath}")
                        continue

                path = f"/posts/{post_id}.json"

                if download:
                    url = self._build_url(path)
                    self._log("GET", url)
                    if self._rate_limiter:
                        self._rate_limiter.wait()
                    resp = self.session.get(url)
                    resp.raise_for_status()
                    data = json.loads(resp.content)

                if save_dir:
                    ext = 'json'
                    filepath = os.path.join(save_dir, f"topic_{topic_id}_post_{post_id}.{ext}")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
                    self._log(f"Saved {filepath}")
                if self.opts.on_post:
                    self.opts.on_post(topic_id, post_id, data)
                if not save_dir:
                    print(data)
            finally:
                self._task_q.task_done()

        self._log(f"Saved {filepath}")

    def download_topic(self, topic_id: int):
        """
        Download all posts in `topic_id`, using a pool of threads.
        """
        if self.threads == []:
            self._start_threads()

        # Fetch post list
        path = f"/t/{topic_id}.json?include_raw=true"
        url  = self._build_url(path)
        # self._log("GET", url)
        if self._rate_limiter:
            self._rate_limiter.wait()
        resp = self.session.get(url)
        resp.raise_for_status()
        data = json.loads(resp.content)
        post_ids = [int(x) for x in data.get("post_stream", {}).get("stream", [])]

        # Determine per-topic save directory
        save_dir = None
        if self.opts.save_path:
            save_dir = os.path.join(self.opts.save_path, f"topic_{topic_id}")
            os.makedirs(save_dir, exist_ok=True)

        # Enqueue
        handled = set()
        for post in data.get("post_stream", {}).get("posts", []):
            self._task_q.put( (topic_id, save_dir, post))
            handled.add(post.get('id', 0))

        for pid in post_ids:
            if pid in handled:
               continue
            self._task_q.put((topic_id, save_dir, pid))

        self._update_index(topic_id)
        # self._log(f"Topic {topic_id} has {len(post_ids)} posts. All posts queued for download for topic.")

    def _update_index(self, topic_id: int):
        try:
            index_path = os.path.join(self.opts.save_path, 'index.json')
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as idxf:
                    index = json.load(idxf)
            else:
                index = {
                    'domain': f"https://{self.host}",
                    'topics': {}
                }

            if index.get('domain') != f"https://{self.host}":
                raise ValueError(f"Index domain mismatch: {index.get('domain')} != https://{self.host}")

            index_key = f"topic_{topic_id}"
            index['topics'][index_key] = datetime.utcnow().isoformat() + 'Z'
            with open(index_path, 'w', encoding='utf-8') as idxf:
                json.dump(index, idxf, indent=2)
            # self._log(f"Updated index file at {index_path}")
        except Exception as e:
            self._log(f"Failed to update index.json: {e}")

    def get_categories(self, subcategories: bool = False) -> list:
        """
        Fetch all top-level categories, then optionally their subcategories.
        """
        # Top-level
        url = self._build_url('/categories.json')
        self._log("GET", url)
        if self._rate_limiter:
            self._rate_limiter.wait()
        resp = self.session.get(url)
        resp.raise_for_status()
        data = json.loads(resp.content)
        top_cats = data.get('category_list', {}).get('categories', [])
        self._log(f"Found {len(top_cats)} top-level categories")

        if not subcategories:
            return top_cats

        # Fetch each subcategory via show endpoint
        self._log(f"Fetching subcategories for {len(top_cats)} categories")
        for cat in top_cats:
            cat['subcategories'] = []
            for sub_id in cat.get('subcategory_ids', []):
                show_url = self._build_url(f'/c/{sub_id}/show.json')
                self._log("GET", show_url)
                if self._rate_limiter:
                    self._rate_limiter.wait()
                r2 = self.session.get(show_url)
                r2.raise_for_status()
                details = json.loads(r2.content)
                cat_info = details.get('category', {})
                cat['subcategories'].append(cat_info)

        return top_cats