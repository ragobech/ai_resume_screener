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
    "Query: Given the above context, you need to evaluate the candidate's resume. Please follow the steps below to reason through your evaluation and identify potential risks.\n"
    "\n"
    "1. **Technical Skills (0-5)**\n"
    "   - **Description:** Assess the candidate's technical proficiency based on the skills listed on their resume. Consider the relevance, depth, and breadth of the technical skills.\n"
    "   - **Chain of Thought:** List the technical skills mentioned. Evaluate any gaps in crucial skills needed for the role. Note any missing key skills that could be a risk.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Detail any skill gaps or areas where the candidate lacks expertise that is critical for the role.\n"
    "\n"
    "2. **Experience and Achievements (0-5)**\n"
    "   - **Description:** Evaluate the candidate's professional experience and accomplishments. Look for inconsistencies, job-hopping, or lack of relevant experience.\n"
    "   - **Chain of Thought:** Examine the work history for frequent changes or gaps. Assess how well the candidate's experience aligns with the role's requirements. Identify any risks associated with their employment history.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Explain any concerns regarding job stability or relevant experience.\n"
    "\n"
    "3. **Education and Certifications (0-5)**\n"
    "   - **Description:** Assess the candidate’s educational background and certifications. Verify the legitimacy and relevance of their credentials.\n"
    "   - **Chain of Thought:** Review educational qualifications and certifications. Check for any unverifiable credentials or degrees from questionable sources. Note any risks associated with their educational background.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Provide a justification for the score based on the verifiability and relevance of the candidate’s credentials.\n"
    "\n"
    "4. **Soft Skills and Leadership (0-5)**\n"
    "   - **Description:** Evaluate the candidate's soft skills and leadership qualities based on their resume. Identify any shortcomings in essential soft skills.\n"
    "   - **Chain of Thought:** Look for evidence of soft skills and leadership roles. Assess if these are adequately demonstrated and if there are any risks related to their soft skills.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Explain how the lack of demonstrated soft skills or leadership qualities might affect the candidate's suitability.\n"
    "\n"
    "5. **Cultural Fit and Alignment with Role (0-5)**\n"
    "   - **Description:** Assess how well the candidate’s background and experience align with the company’s culture and the specific role they are applying for.\n"
    "   - **Chain of Thought:** Evaluate if the candidate’s values and background align with the company’s culture. Identify any risks related to potential cultural misfit or role alignment.\n"
    "   - **Score:** [0-5]\n"
    "   - **Explanation:** Justify the score based on the alignment with the company’s culture and role expectations. Note any risks related to cultural fit.\n"
    "\n"
    "6. **Additional Risks:**\n"
    "   - **Description:** Identify any other risks that may not fall into the above categories but are relevant to the candidate’s suitability for the role.\n"
    "   - **Chain of Thought:** Consider any other potential red flags such as exaggerated claims or unverifiable information. Assess any additional risks and their impact on the candidate's evaluation.\n"
    "   - **Explanation:** Provide a detailed explanation of any additional risks identified and their implications.\n"
    "\n"
    "Finally, provide an overall summary of the candidate’s evaluation based on the above criteria and identified risks.\n"
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