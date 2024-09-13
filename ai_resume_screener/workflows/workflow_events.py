from llama_index.core.workflow import Event


class IntervieweeResponseEvent(Event):
    response: str
    summary: str


class IntervieweeSummaryEvent(Event):
    query: str
    response: str
    summary: str
