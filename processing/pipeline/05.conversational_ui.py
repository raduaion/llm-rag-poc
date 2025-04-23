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
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import common.general_config as gcfg
import gradio as gr
import sys
import time
import warnings
import base64

import common.conversational_pipeline as cp
from common.conversational_pipeline import IDX_KEY, SECONDARY_IDX_KEY
import ui.HermesChatInterface as hci

start = time.perf_counter_ns()
cp.ConversationalPipeline.initialize()
end = time.perf_counter_ns()
gcfg.llm_log(f"Load time: ({(end - start) // 1000000000}s): ")

def chat_request(message, history, index_list, secondary_index_list, all_indexes, clear, logout, request: gr.Request):
    selected = [ all_indexes[idx] for idx in index_list]
    selected_secondary = [ all_indexes[idx] for idx in secondary_index_list]
    if len(selected) == 0:
        gr.Warning("Cannot query Hermes. Please select one or more indexes.")
        return "Please select one or more indexes."

    session = cp.ConversationalPipeline.get(indexes={ IDX_KEY: selected, SECONDARY_IDX_KEY: selected_secondary }, request=request)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter_ns()
        gcfg.llm_log("", session)
        gcfg.start(session)
        result = cp.ConversationalPipeline.call_rag(message, session)
        # result = {'answer': "Removed for testing", 'question': message}
        gcfg.stop(session)
        end = time.perf_counter_ns()
        gcfg.llm_log(f"[#{session['question_number']}] You asked: {message}", session)
        gcfg.llm_log("---------------", session)
        gcfg.llm_log(f"[#{session['question_number']}] Contextualized question: " + result['question'], session)
        gcfg.llm_log("", session)
        gcfg.llm_log("---------------", session)
        gcfg.llm_log(f"[#{session['question_number']}] Answer ({(end - start) // 1000000000}s): " + result['answer'], session)
        gcfg.llm_log("", session)

        return f"Contextualized question: {result['question']}\n\n[#{session['question_number']}]({(end - start) // 1000000000}s) " + result['answer']

def check_user(all_indexes, request: gr.Request):
    username = "anonymous"
    if request:
        if 'authorization' in request.headers:
            auth = request.headers['authorization']
            username = base64.b64decode(request.headers['authorization'].removeprefix('Basic ')).decode('utf-8').split(':')[0]
    gcfg.llm_log(f"System loaded for user {username}")
    session = cp.ConversationalPipeline.get(request=request, create = False)

    clear_text = "Clear session"
    index_list = [ gcfg.newest_index(gcfg.INDEXES_FOLDER)["name"] ]
    secondary_index_list = []
    if not session is None:
        indexes = session['indexes'][IDX_KEY]
        secondary_indexes = session['indexes'][SECONDARY_IDX_KEY]
        index_list = [key for key, value in all_indexes.items() if value in indexes]
        secondary_index_list = [key for key, value in all_indexes.items() if value in secondary_indexes]
        gcfg.llm_log(f"Session number of questions = {session['question_number']}")
        gr.Warning(f"Using existing session with {session['question_number']} questions, primary idx={index_list}, secondary idx={secondary_index_list}")
        clear_text = clear_text + f" [{session['question_number']} question(s)]"
    return gr.update(value = f"Logout ({username})"), gr.update(value = index_list), gr.update(value = secondary_index_list), gr.update(value = clear_text)

def clear_session(request: gr.Request):
    result = 0
    if request:
        result = cp.ConversationalPipeline.clear(request)

    if result == 0:
        gr.Warning("No session to clear, ask some questions.")
    else:
        gr.Info("The session was cleared.")
    return [], [], gr.update(value = "Clear session")

def changed_index_primary(index_list, secondary_index_list, all_indexes, request: gr.Request):
    return changed_index(IDX_KEY, index_list, secondary_index_list, all_indexes, request)

def changed_index_secondary(index_list, secondary_index_list, all_indexes, request: gr.Request):
    return changed_index(SECONDARY_IDX_KEY, index_list, secondary_index_list, all_indexes, request)

def changed_index(session_key, index_list, secondary_index_list, all_indexes, request: gr.Request):
    gcfg.llm_log(f"Indexes '{session_key}' changed to '{index_list}'")
    values = check_indexes(index_list, secondary_index_list, all_indexes, request)
    clear_text = values[0]
    result = values[1]
    if result:
        result = cp.ConversationalPipeline.clear(request)

    if result != 0:
        gr.Info("The session was cleared.")
    return [], [],  gr.update(value = clear_text)

def changed_chat(index_list, secondary_index_list, all_indexes, request: gr.Request):
    values = check_indexes(index_list, secondary_index_list, all_indexes, request)
    clear_text = values[0]
    return gr.update(value = clear_text)

def check_indexes(index_list, secondary_index_list, all_indexes, request):
    clear_text = "Clear session"
    result = 0
    if request:
        selected_primary = [ all_indexes[idx] for idx in index_list]
        selected_secondary = [ all_indexes[idx] for idx in secondary_index_list]
        session = cp.ConversationalPipeline.get(indexes = { IDX_KEY: selected_primary, SECONDARY_IDX_KEY: selected_secondary }, request = request, create = False)
        if not session is None:
            selected_primary.sort()
            session['indexes'][IDX_KEY].sort()
            selected_secondary.sort()
            session['indexes'][SECONDARY_IDX_KEY].sort()
            if session['indexes'][IDX_KEY] == selected_primary and session['indexes'][SECONDARY_IDX_KEY] == selected_secondary:
                clear_text = clear_text + f" [{session['question_number']} question(s)]"
            else:
                result = 1
    return clear_text, result


logout_button = gr.Button(value = "Logout", link = "/logout")
clear_button = gr.Button(value = "Clear session")
chatbot = gr.Chatbot(height=500, layout='panel', label="Hermes Assistant", show_copy_button=True, render_markdown=False, type="messages")
textbox = gr.Textbox(placeholder="Ask me a question", container=False, scale=7)

all = gcfg.get_indexes(gcfg.INDEXES_FOLDER)
indexes = { all[x]["name"]: all[x]["path"] for x in all }
index_dropdown = gr.Dropdown(indexes, label="Primary Index", info="Select the index to use for primary context (changing this will clear the current session)", multiselect=True)
secondary_index_dropdown = gr.Dropdown(indexes, label="Secondary Index", info="Select the index to use for secondary context (changing this will clear the current session)", multiselect=True)
# print(indexes)
# print(gcfg.newest_index(gcfg.INDEXES_FOLDER)["name"])
with gr.Blocks(title = "Hermes AI", analytics_enabled=False, css="footer{display:none !important}") as hermes:
    all_indexes = gr.State(indexes)
    chat = hci.HermesChatInterface(
        chat_request,
        changed_chat,
        [index_dropdown, secondary_index_dropdown, all_indexes],
        [clear_button],
        chatbot=chatbot,
        textbox=textbox,
        description=f"Ask any question",
        theme="soft",
        additional_inputs = [index_dropdown, secondary_index_dropdown, all_indexes, clear_button, logout_button],
        additional_inputs_accordion = gr.Accordion(label="Manage Session", open=True, render=False),
        type="messages"
    )
    clear_button.click(clear_session, [], outputs=[chatbot, chat.chatbot_state, clear_button])
    hermes.load(check_user, [all_indexes], [logout_button, index_dropdown, secondary_index_dropdown, clear_button])
    index_dropdown.change(changed_index_primary, [index_dropdown, secondary_index_dropdown, all_indexes], outputs=[chatbot, chat.chatbot_state, clear_button])
    secondary_index_dropdown.change(changed_index_secondary, [index_dropdown, secondary_index_dropdown, all_indexes], outputs=[chatbot, chat.chatbot_state, clear_button])

hermes.queue().launch(favicon_path = 'favicon.ico', server_name='0.0.0.0', debug=True)
