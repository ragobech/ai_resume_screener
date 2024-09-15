import unittest
from unittest.mock import patch, AsyncMock, mock_open, MagicMock
from ai_resume_screener.workflows.core.screening_core import ScreeningCore
from ai_resume_screener.workflows.workflow_events import IntervieweeResponseEvent
from ai_resume_screener.workflows.resume_screening_agent import ResumeScreeningAgent

class TestResumeScreeningAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = ResumeScreeningAgent()

    @patch('resume_screening_agent.ScreeningCore')
    @patch('resume_screening_agent.OpenAI')
    async def test_pre_process(self, mock_openai, mock_screening_core):
        # Arrange
        mock_screening_core_instance = mock_screening_core.return_value
        mock_screening_core_instance.retriever_query_engine.return_value.aquery = AsyncMock(return_value="mocked response")
        mock_openai.return_value = MagicMock()

        mock_event = MagicMock()
        mock_event.get.return_value = "Evaluate the resume of the candidate for a VP of Engineering position."

        # Act
        result = await self.agent.pre_process(ctx={}, ev=mock_event)

        # Assert
        mock_screening_core.assert_called_once_with(candidate_doc='chris.pdf')
        mock_screening_core_instance.retriever_query_engine().aquery.assert_awaited_once_with("Evaluate the resume of the candidate for a VP of Engineering position.")
        self.assertIsInstance(result, IntervieweeResponseEvent)
        self.assertEqual(result.response, "mocked response")

    @patch('resume_screening_agent.OpenAI')
    async def test_prepare_summary(self, mock_openai):
        # Arrange
        mock_llm_instance = mock_openai.return_value
        mock_llm_instance.acomplete = AsyncMock(return_value="mocked summary response")

        mock_event = IntervieweeResponseEvent(response="mock response")
        ctx = {"user_query": "mock user query"}

        # Act
        result = await self.agent.prepare_summary(ctx=ctx, ev=mock_event)

        # Assert
        mock_llm_instance.acomplete.assert_awaited_once()
        self.assertIsInstance(result, IntervieweeResponseEvent)
        self.assertEqual(result.summary, "mocked summary response")
        self.assertEqual(result.response, "mock response")
        self.assertEqual(result.query, "mock user query")

    @patch('builtins.open', new_callable=mock_open)
    async def test_save_summary(self, mock_file):
        # Arrange
        mock_event = IntervieweeResponseEvent(
            summary="mocked summary",
            response="mocked response",
            query="mocked query"
        )
        ctx = {"user_query": "mock user query"}

        with patch('builtins.input', return_value="mock feedback"):
            # Act
            result = await self.agent.save_summary(ctx=ctx, ev=mock_event)

        # Assert
        mock_file.assert_called_with('./chris.md', mode='w')
        mock_file().write.assert_any_call('user_query : mocked query\n')
        mock_file().write.assert_any_call('agent_response : mocked response\n')
        mock_file().write.assert_any_call('summary : mocked summary\n')

        mock_file.assert_called_with('./chris_feedback.md', mode='w')
        mock_file().write.assert_called_with('feedback : mock feedback\n')

        self.assertEqual(result.result, "Summary: mocked summary\nFeedback: mock feedback")

if __name__ == '__main__':
    unittest.main()