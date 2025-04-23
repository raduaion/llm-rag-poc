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
import sys
import time
import warnings

import common.conversational_pipeline as cp

start = time.perf_counter_ns()
cp.ConversationalPipeline.initialize()
end = time.perf_counter_ns()
gcfg.llm_log(f"Load time: ({(end - start) // 1000000000}s): ")

session = cp.ConversationalPipeline.get()
query = ""
while query != gcfg.EXIT_COMMAND:
    gcfg.llm_log("-------------------------------------------------")
    try:
        query = input(f"[#{session['question_number']}] Please enter your question: ")
        if query == gcfg.EXIT_COMMAND:
            continue
        if query == gcfg.CLEAR_COMMAND:
            cp.ConversationalPipeline.clear()
            gcfg.llm_log(f"Session with {session['question_number']} questions was cleared")
            session = cp.ConversationalPipeline.get()
            continue
    except KeyboardInterrupt:
        gcfg.llm_log(f"\n{gcfg.EXIT_COMMAND}")
        sys.exit(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter_ns()
        gcfg.llm_log("", session)
        gcfg.start(session)
        result = cp.ConversationalPipeline.call_rag(query, session)
        # result = {'answer': "Removed for testing", 'question': message}
        gcfg.stop(session)
        end = time.perf_counter_ns()
        gcfg.llm_log(f"[#{session['question_number']}] You asked: {query}")
        gcfg.llm_log("---------------")
        gcfg.llm_log(f"[#{session['question_number']}] Contextualized question: " + result['question'])
        gcfg.llm_log("", session)
        gcfg.llm_log("---------------")
        gcfg.llm_log(f"[#{session['question_number']}] Answer ({(end - start) // 1000000000}s): " + result['answer'])
        gcfg.llm_log("")
