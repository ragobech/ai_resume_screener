import os
import openai
import logging
from typing import Any
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings, PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step
)
from ai_resume_screener.workflows.workflow_events import IntervieweeResponseEvent
from ai_resume_screener.workflows.core.screening_core import ScreeningCore
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class ResumeScreeningAgent(Workflow):
    def __init__(
            self,
            *args: Any,
            llm: LLM | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm or OpenAI(model='gpt-4', request_timeout=300)
        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        Settings.callback_manager = callback_manager
        self.file_name = 'chris.pdf'

    @step(pass_context=True)
    async def pre_process(self, ctx: Context, ev: StartEvent) -> IntervieweeResponseEvent:
        try:
            user_query = ev.get("user_query")
            fa = ScreeningCore(candidate_doc=self.file_name)
            ctx.data['user_query'] = user_query
            response = await fa.retriever_query_engine().aquery(user_query)
            logging.info(f'response from llm: {str(response)}')
            return IntervieweeResponseEvent(response=str(response))
        except Exception as e:
            logging.error(str(e))

    @step(pass_context=True)
    async def prepare_summary(self, ctx: Context, ev: IntervieweeResponseEvent) -> IntervieweeResponseEvent:
        try:
            # get chat context and response
            current_query = ctx.data.get("user_query", [])
            current_context = ev.response
            prompt_tmpl_str = (
                "---------------------\n"
                f"{current_context}\n"
                "---------------------\n"
                "Query: Given the above context, summarize the candidate report with Key Highlights "
                "and important Performance Indicators.\n"
                "Create an Evaluation of the candidate's resume based on the following four categories."
                "For each category, provide a score from 0 to 5, where 0 is the lowest and 5 is the highest. Include a brief explanation for each score.\n"
               
                "Technical Skills (0-5):\n"
                "Description: Assess the candidate's technical proficiency based on the skills listed on their resume. Consider the relevance, depth, and breadth of the technical skills.\n"
                "Score: [0-5]\n"
                "Explanation:\n"

                "Experience and Achievements (0-5):\n"
                "Description: Evaluate the candidate's professional experience and accomplishments. Consider the significance of their roles, responsibilities, and the impact of their achievements.\n"
                "Score: [0-5]\n"
                "Explanation:\n"

                "Education and Certifications (0-5):\n"
                "Description: Assess the candidate’s educational background and any relevant certifications. Consider the relevance of their education and any additional qualifications that support their expertise.\n"
                "Score: [0-5]\n"
                "Explanation:\n"

                "Soft Skills and Leadership (0-5):\n"
                "Description: Evaluate the candidate's soft skills and leadership qualities based on their resume. Consider their ability to work in teams, communication skills, and leadership experience.\n"
                "Score: [0-5]\n"
                "Explanation:\n"

                "Cultural Fit and Alignment with Role (0-5):\n"
                "Description: Assess how well the candidate’s background and experience align with the company’s culture and the specific role they are applying for. Consider their potential to fit in with the team and contribute to the company's goals.\n"
                "Score: [0-5]\n"
                "Explanation:\n"
                "Answer: "
            )
            prompt_tmpl = PromptTemplate(prompt_tmpl_str)
            summary_response = await self.llm.acomplete(prompt_tmpl_str)
            return IntervieweeResponseEvent(summary=str(summary_response), response=str(current_context), query=str(current_query))
        except Exception as e:
            logging.error(str(e))

    @step(pass_context=True)
    async def save_summary(self, ctx: Context, ev: IntervieweeResponseEvent) -> StopEvent:
        try:
            current_query = ctx.data.get('user_query')
            current_response = ev.response
            current_summary = ev.summary

            # Save summary to file
            with open(f'./{self.file_name.strip(".pdf")}.md', mode='w') as script:
                script.write(f'user_query : {current_query}\n')
                script.write(f'agent_response : {current_response}\n')
                script.write(f'summary : {current_summary}\n')

            # Prompt user for feedback
            feedback = input("Please provide feedback on the evaluation (e.g., accuracy, relevancy, score changes): ")

            # Save feedback
            with open(f'./{self.file_name.strip(".pdf")}_feedback.md', mode='w') as feedback_file:
                feedback_file.write(f'feedback : {feedback}\n')

            # Return the result and feedback
            return StopEvent(result=f"Summary: {current_summary}\nFeedback: {feedback}")

        except Exception as e:
            logging.error(str(e))