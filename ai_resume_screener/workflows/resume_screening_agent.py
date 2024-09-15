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
    "Query: Given the above context,summarize the candidate report with Key Highlights "
                "and important Performance Indicators.\n"
    "Create an Evaluation of the candidate's resume based on the following categories. Please follow the steps below to reason through your evaluation and provide a summary.\n"
    "\n"
    "1. **Technical Skills (0-5)**\n"
    "   - **Description:** Assess the candidate's technical proficiency based on the skills listed on their resume. Consider the relevance, depth, and breadth of the technical skills.\n"
    "   - **Chain of Thought:** Start by listing the technical skills mentioned in the resume. Evaluate each skill's relevance to the role, depth (e.g., level of expertise), and breadth (e.g., variety of skills). Use this evaluation to determine a score from 0 to 5.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Provide a detailed explanation of how you arrived at the score, including any specific skills that stood out or any gaps noticed.\n"
    "\n"
    "2. **Experience and Achievements (0-5)**\n"
    "   - **Description:** Evaluate the candidate's professional experience and accomplishments. Consider the significance of their roles, responsibilities, and the impact of their achievements.\n"
    "   - **Chain of Thought:** List key positions held and significant achievements. Assess the impact of these roles on their professional development and their relevance to the applied role. Use this to determine a score from 0 to 5.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Explain the rationale behind the score, highlighting important roles and achievements that influenced your evaluation.\n"
    "\n"
    "3. **Education and Certifications (0-5)**\n"
    "   - **Description:** Assess the candidate’s educational background and any relevant certifications. Consider the relevance of their education and any additional qualifications that support their expertise.\n"
    "   - **Chain of Thought:** Review the candidate’s educational qualifications and certifications. Evaluate how these credentials support their ability to perform in the role. Score based on the relevance and quality of education and certifications.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Provide a justification for the score, noting any particularly relevant degrees or certifications.\n"
    "\n"
    "4. **Soft Skills and Leadership (0-5)**\n"
    "   - **Description:** Evaluate the candidate's soft skills and leadership qualities based on their resume. Consider their ability to work in teams, communication skills, and leadership experience.\n"
    "   - **Chain of Thought:** Identify any mentioned soft skills and leadership experiences. Assess how these qualities contribute to their suitability for the role. Determine a score from 0 to 5 based on your assessment.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Explain how the soft skills and leadership qualities influenced the score, citing examples from the resume if applicable.\n"
    "\n"
    "5. **Cultural Fit and Alignment with Role (0-5)**\n"
    "   - **Description:** Assess how well the candidate’s background and experience align with the company’s culture and the specific role they are applying for. Consider their potential to fit in with the team and contribute to the company's goals.\n"
    "   - **Chain of Thought:** Evaluate how the candidate’s background aligns with the company's culture and the role’s requirements. Consider any indicators of cultural fit or misalignment. Score from 0 to 5 based on this alignment.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Justify the score based on how well the candidate’s profile matches the company’s culture and role expectations.\n"
    "\n"
    "Finally, provide an overall summary of the candidate’s evaluation based on the above criteria.\n"
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