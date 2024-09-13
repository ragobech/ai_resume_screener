
from ai_resume_screener.workflows.resume_screening_agent import ResumeScreeningAgent
import nest_asyncio


# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def main():
    screen = ResumeScreeningAgent(timeout=300, verbose=True)
    user_query = ("What are the main details about this candidate?")
    candidate_assessment= await screen.run(user_query=user_query)
    print(candidate_assessment)
    

if __name__ == '__main__':
    import asyncio

    asyncio.run(main=main())