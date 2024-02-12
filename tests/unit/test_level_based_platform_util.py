import unittest
from llama_index.schema import NodeWithScore, TextNode
from utils.query_engine.level_based_platforms_util import LevelBasedPlatformUtils


class TestLevelBasedPlatformUtils(unittest.TestCase):
    def setUp(self):
        self.level1_key = "channel"
        self.level2_key = "thread"
        self.date_key = "date"
        self.utils = LevelBasedPlatformUtils(
            self.level1_key, self.level2_key, self.date_key
        )

    def test_prepare_prompt_with_metadata_info(self):
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="content1",
                    metadata={"author_username": "user1", "date": "2022-01-01"},
                ),
                score=0,
            ),
            NodeWithScore(
                node=TextNode(
                    text="content2",
                    metadata={"author_username": "user2", "date": "2022-01-02"},
                ),
                score=0,
            ),
        ]
        prefix = "  "
        expected_output = (
            "  author: user1\n  message_date: 2022-01-01\n  message 1: content1\n"
            "  author: user2\n  message_date: 2022-01-02\n  message 2: content2"
        )
        result = self.utils.prepare_prompt_with_metadata_info(nodes, prefix)
        self.assertEqual(result, expected_output)

    def test_group_nodes_per_metadata(self):
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="content1",
                    metadata={"channel": "A", "thread": "X", "date": "2022-01-01"},
                ),
                score=0,
            ),
            NodeWithScore(
                node=TextNode(
                    text="content2",
                    metadata={"channel": "A", "thread": "Y", "date": "2022-01-01"},
                ),
                score=0,
            ),
            NodeWithScore(
                node=TextNode(
                    text="content3",
                    metadata={"channel": "B", "thread": "X", "date": "2022-01-02"},
                ),
                score=0,
            ),
        ]
        expected_output = {
            "A": {"X": {"2022-01-01": [nodes[0]]}, "Y": {"2022-01-01": [nodes[1]]}},
            "B": {"X": {"2022-01-02": [nodes[2]]}},
        }
        result = self.utils.group_nodes_per_metadata(nodes)
        self.assertEqual(result, expected_output)

    def test_prepare_context_str_based_on_summaries(self):
        raw_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="raw_content1",
                    metadata={
                        "channel": "A",
                        "thread": "X",
                        "date": "2022-01-01",
                        "author_username": "USERNAME#1",
                    },
                ),
                score=0,
            ),
            NodeWithScore(
                node=TextNode(
                    text="raw_content2",
                    metadata={
                        "channel": "A",
                        "thread": "Y",
                        "date": "2022-01-04",
                        "author_username": "USERNAME#2",
                    },
                ),
                score=0,
            ),
        ]
        summary_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="summary_content",
                    metadata={"channel": "A", "thread": "X", "date": "2022-01-01"},
                ),
                score=0,
            )
        ]
        grouped_raw_nodes = {"A": {"X": {"2022-01-01": raw_nodes}}}
        grouped_summary_nodes = {"A": {"X": {"2022-01-01": summary_nodes}}}
        expected_output = (
            """channel: A\nthread: X\ndate: 2022-01-01\nsummary: summary_content\nmessages:\n"""
            """  author: USERNAME#1\n  message_date: 2022-01-01\n  message 1: raw_content1\n"""
            """  author: USERNAME#2\n  message_date: 2022-01-04\n  message 2: raw_content2\n"""
        )
        result, _ = self.utils.prepare_context_str_based_on_summaries(
            grouped_raw_nodes, grouped_summary_nodes
        )
        self.assertEqual(result.strip(), expected_output.strip())

    def test_prepare_context_str_based_on_summaries_no_summary(self):
        node1 = NodeWithScore(
            node=TextNode(
                text="raw_content1",
                metadata={
                    "channel": "A",
                    "thread": "X",
                    "date": "2022-01-01",
                    "author_username": "USERNAME#1",
                },
            ),
            score=0,
        )
        node2 = NodeWithScore(
            node=TextNode(
                text="raw_content2",
                metadata={
                    "channel": "A",
                    "thread": "Y",
                    "date": "2022-01-04",
                    "author_username": "USERNAME#2",
                },
            ),
            score=0,
        )
        grouped_raw_nodes = {
            "A": {"X": {"2022-01-01": [node1]}, "Y": {"2022-01-04": [node2]}}
        }
        grouped_summary_nodes = {}
        result, (
            summary_nodes_to_fetch_filters,
            raw_nodes_missed,
        ) = self.utils.prepare_context_str_based_on_summaries(
            grouped_raw_nodes, grouped_summary_nodes
        )
        self.assertEqual(result, "")
        self.assertEqual(len(summary_nodes_to_fetch_filters), 2)
        for channel in raw_nodes_missed.keys():
            self.assertIn(channel, ["A"])
            for thread in raw_nodes_missed[channel].keys():
                self.assertIn(thread, ["X", "Y"])
                for date in raw_nodes_missed[channel][thread]:
                    self.assertIn(date, ["2022-01-01", "2022-01-04"])
                    nodes = raw_nodes_missed[channel][thread][date]

                    if date == "2022-01-01":
                        self.assertEqual(
                            grouped_raw_nodes["A"]["X"]["2022-01-01"], nodes
                        )
                    elif date == "2022-01-04":
                        self.assertEqual(
                            grouped_raw_nodes["A"]["Y"]["2022-01-04"], nodes
                        )

    def test_prepare_context_str_based_on_summaries_multiple_summaries_error(self):
        raw_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="raw_content1",
                    metadata={"channel": "A", "thread": "X", "date": "2022-01-01"},
                ),
                score=0,
            ),
            NodeWithScore(
                node=TextNode(
                    text="raw_content2",
                    metadata={"channel": "A", "thread": "Y", "date": "2022-01-01"},
                ),
                score=0,
            ),
        ]
        summary_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="summary_content1",
                    metadata={"channel": "A", "thread": "X", "date": "2022-01-01"},
                ),
                score=0,
            ),
            NodeWithScore(
                node=TextNode(
                    text="summary_content2",
                    metadata={"channel": "A", "thread": "X", "date": "2022-01-01"},
                ),
                score=0,
            ),
        ]
        grouped_raw_nodes = {"A": {"X": {"2022-01-01": raw_nodes}}}
        grouped_summary_nodes = {"A": {"X": {"2022-01-01": summary_nodes}}}
        with self.assertRaises(ValueError):
            self.utils.prepare_context_str_based_on_summaries(
                grouped_raw_nodes, grouped_summary_nodes
            )
