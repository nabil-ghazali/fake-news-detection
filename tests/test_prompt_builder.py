import unittest
from unittest.mock import patch

from prompt.prompt_builder import PromptBuilder


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.article_text = "Donald Trump shares a controversial photo on social media."
        self.model_embedding = "all-minilm:latest"
        self.model_llm = "phi3:3.8b"
        self.builder = PromptBuilder(
            self.article_text, self.model_embedding, self.model_llm
        )

    def test_build_context_for_prompt(self):
        search_results = {
            "documents": [["Doc 1 text", "Doc 2 text"]],
            "metadatas": [
                [
                    {"subject": "Politics", "date": "2025-10-27", "label": "True"},
                    {"subject": "Media", "date": "2025-10-26", "label": "Fake"},
                ]
            ],
        }
        context = self.builder.build_context_for_prompt(search_results)
        self.assertIn("Doc 1 text", context)
        self.assertIn("Politics", context)
        self.assertIn("Fake", context)

    def test_build_prompt(self):
        context = "Some context text"
        prompt = self.builder.build_prompt(context)
        self.assertIn("Some context text", prompt)
        self.assertIn(self.article_text, prompt)
        self.assertIn("Label", prompt)

    @patch("prompt.prompt_builder.ollama.Client")
    def test_predict_label(self, mock_client_cls):
        # Le code appelle ollama.Client(host=...).generate(...) : on mocke l'instance.
        mock_client = mock_client_cls.return_value
        mock_client.generate.return_value = {
            "response": "Label: True\nJustification: Example."
        }

        response = self.builder.predict_label("Test prompt")

        self.assertEqual(response, "Label: True\nJustification: Example.")
        mock_client.generate.assert_called_once_with(
            model=self.model_llm, prompt="Test prompt"
        )
