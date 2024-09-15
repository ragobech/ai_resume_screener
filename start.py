
from ai_resume_screener.workflows.resume_screening_agent import ResumeScreeningAgent
import nest_asyncio


# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def main():
    agent = ResumeScreeningAgent(timeout=300, verbose=True)
    user_query = ("Evaluate the resume of the candidate for a VP of Engineering position.")
    candidate_assessment= await agent.run(user_query=user_query)
    print(candidate_assessment)
    

if __name__ == '__main__':
    import asyncio

    asyncio.run(main=main())