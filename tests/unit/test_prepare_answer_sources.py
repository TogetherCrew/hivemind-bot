import unittest

from llama_index.core.query_engine import SubQuestionAnswerPair
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.schema import NodeWithScore, TextNode
from utils.query_engine.prepare_answer_sources import PrepareAnswerSources


class TestPrepareAnswerSources(unittest.TestCase):
    def setUp(self) -> None:
        self.prepare = PrepareAnswerSources(threshold=0.7)

    def test_empty_nodes_list(self):
        """Test with an empty list of nodes."""
        nodes = []
        result = self.prepare.prepare_answer_sources(nodes)
        self.assertEqual(result, "")

    def test_single_tool_with_high_score_urls(self):
        """Test with a single tool containing multiple URLs with scores above threshold."""
        node1 = NodeWithScore(
            node=TextNode(
                text="content 1", metadata={"url": "https://github.com/repo1"}
            ),
            score=0.8,
        )
        node2 = NodeWithScore(
            node=TextNode(
                text="content 2", metadata={"url": "https://github.com/repo2"}
            ),
            score=0.9,
        )

        nodes = [
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="github", sub_question="Question"),
                sources=[node1, node2],
            )
        ]
        result = self.prepare.prepare_answer_sources(nodes)
        expected = (
            "References:\n"
            "github:\n"
            "[1] https://github.com/repo1\n"
            "[2] https://github.com/repo2"
        )
        self.assertEqual(result, expected)

    def test_urls_below_score_threshold(self):
        """Test with URLs that have scores below the 0.7 threshold."""
        node1 = NodeWithScore(
            node=TextNode(
                text="content 1", metadata={"url": "https://github.com/repo1"}
            ),
            score=0.6,
        )
        node2 = NodeWithScore(
            node=TextNode(
                text="content 2", metadata={"url": "https://github.com/repo2"}
            ),
            score=0.5,
        )

        nodes = [
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="github", sub_question="Question"),
                sources=[node1, node2],
            )
        ]
        result = self.prepare.prepare_answer_sources(nodes)
        self.assertEqual(result, "")

    def test_mixed_score_urls(self):
        """Test with a mixture of high and low score URLs."""
        nodes = [
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="github", sub_question="Question"),
                sources=[
                    NodeWithScore(
                        node=TextNode(
                            text="content 1",
                            metadata={"url": "https://github.com/repo1"},
                        ),
                        score=0.8,
                    ),
                    NodeWithScore(
                        node=TextNode(
                            text="content 2",
                            metadata={"url": "https://github.com/repo2"},
                        ),
                        score=0.6,  # Below threshold
                    ),
                    NodeWithScore(
                        node=TextNode(
                            text="content 3",
                            metadata={"url": "https://github.com/repo3"},
                        ),
                        score=0.9,
                    ),
                ],
            )
        ]
        result = self.prepare.prepare_answer_sources(nodes)
        expected = (
            "References:\n"
            "github:\n"
            "[1] https://github.com/repo1\n"
            "[2] https://github.com/repo3"
        )
        self.assertEqual(result, expected)

    def test_multiple_tools_with_valid_scores(self):
        """Test with multiple tools containing URLs with valid scores."""
        nodes = [
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="github", sub_question="Question"),
                sources=[
                    NodeWithScore(
                        node=TextNode(
                            text="content 1",
                            metadata={"url": "https://github.com/repo1"},
                        ),
                        score=0.8,
                    ),
                    NodeWithScore(
                        node=TextNode(
                            text="content 2",
                            metadata={"url": "https://github.com/repo2"},
                        ),
                        score=0.75,
                    ),
                ],
            ),
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="stackoverflow", sub_question="Question"),
                sources=[
                    NodeWithScore(
                        node=TextNode(
                            text="content 3",
                            metadata={"url": "https://stackoverflow.com/q1"},
                        ),
                        score=0.9,
                    ),
                    NodeWithScore(
                        node=TextNode(
                            text="content 4",
                            metadata={"url": "https://stackoverflow.com/q2"},
                        ),
                        score=0.85,
                    ),
                ],
            ),
        ]
        result = self.prepare.prepare_answer_sources(nodes)
        expected = (
            "References:\n"
            "github:\n"
            "[1] https://github.com/repo1\n"
            "[2] https://github.com/repo2\n\n"
            "stackoverflow:\n"
            "[1] https://stackoverflow.com/q1\n"
            "[2] https://stackoverflow.com/q2"
        )
        self.assertEqual(result, expected)

    def test_none_urls_with_valid_scores(self):
        """Test with None URLs that have valid scores."""
        nodes = [
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="github", sub_question="Question"),
                sources=[
                    NodeWithScore(
                        node=TextNode(text="content 1", metadata={"url": None}),
                        score=0.8,
                    ),
                    NodeWithScore(
                        node=TextNode(
                            text="content 2",
                            metadata={"url": "https://github.com/repo2"},
                        ),
                        score=0.9,
                    ),
                ],
            )
        ]
        result = self.prepare.prepare_answer_sources(nodes)
        self.assertEqual(
            result, ("References:\n" "github:\n" "[1] https://github.com/repo2")
        )

    def test_missing_urls_with_valid_scores(self):
        """Test with missing URLs that have valid scores."""
        nodes = [
            SubQuestionAnswerPair(
                sub_q=SubQuestion(tool_name="github", sub_question="Question"),
                sources=[
                    NodeWithScore(
                        node=TextNode(text="content 1", metadata={}), score=0.8
                    ),
                    NodeWithScore(
                        node=TextNode(
                            text="content 2",
                            metadata={"url": "https://github.com/repo2"},
                        ),
                        score=0.9,
                    ),
                ],
            )
        ]
        result = self.prepare.prepare_answer_sources(nodes)
        self.assertEqual(
            result, ("References:\n" "github:\n" "[1] https://github.com/repo2")
        )
