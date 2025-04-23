import gradio as gr
from gradio.events import Events
from gradio.events import Dependency, on
from typing import Callable

class HermesChatInterface(gr.ChatInterface):

    def __init__(self, fn: Callable, change_fn: Callable, change_inputs, change_outputs: list, **kwargs):
        self._hermes_fn = fn
        self._change_fn = change_fn
        self._change_inputs = change_inputs
        self._change_outputs = change_outputs
        bound_callable = (lambda handler, s:
            lambda request, *args : handler(s, *args, request=request)
        )(HermesChatInterface._callable, self)
        bound_callable.__annotations__['request'] = gr.Request
        super().__init__(bound_callable, **kwargs)

    def _callable(self, *args, request: gr.Request):
        # print(f"_callable {self}, args {args} request {request}")
        return self._hermes_fn(*args, request)

    def _setup_events(self):
        super()._setup_events()
        on(
            [self.chatbot.change],
            self._change_fn,
            self._change_inputs,
            self._change_outputs,
            show_api=False,
            queue=False,
        )