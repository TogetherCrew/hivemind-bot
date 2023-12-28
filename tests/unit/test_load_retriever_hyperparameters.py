import unittest
from unittest.mock import patch

from bot.retrievers.utils.load_hyperparams import load_hyperparams


class TestLoadHyperparams(unittest.TestCase):
    @patch("os.getenv")
    def test_valid_hyperparams(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K1_RETRIEVER_SEARCH": "10",
            "K2_RETRIEVER_SEARCH": "20",
            "D_RETRIEVER_SEARCH": "30",
        }.get(x)
        result = load_hyperparams()
        self.assertEqual(result, (10, 20, 30))

    @patch("os.getenv")
    def test_missing_k1(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K2_RETRIEVER_SEARCH": "20",
            "D_RETRIEVER_SEARCH": "30",
        }.get(x)
        with self.assertRaises(ValueError):
            load_hyperparams()

    @patch("os.getenv")
    def test_missing_k2(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K1_RETRIEVER_SEARCH": "10",
            "D_RETRIEVER_SEARCH": "30",
        }.get(x)
        with self.assertRaises(ValueError):
            load_hyperparams()

    @patch("os.getenv")
    def test_missing_d(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K1_RETRIEVER_SEARCH": "10",
            "K2_RETRIEVER_SEARCH": "20",
        }.get(x)
        with self.assertRaises(ValueError):
            load_hyperparams()

    @patch("os.getenv")
    def test_invalid_k1(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K1_RETRIEVER_SEARCH": "invalid",
            "K2_RETRIEVER_SEARCH": "20",
            "D_RETRIEVER_SEARCH": "30",
        }.get(x)
        with self.assertRaises(ValueError):
            load_hyperparams()

    @patch("os.getenv")
    def test_invalid_k2(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K1_RETRIEVER_SEARCH": "10",
            "K2_RETRIEVER_SEARCH": "invalid",
            "D_RETRIEVER_SEARCH": "30",
        }.get(x)
        with self.assertRaises(ValueError):
            load_hyperparams()

    @patch("os.getenv")
    def test_invalid_d(self, mock_getenv):
        mock_getenv.side_effect = lambda x: {
            "K1_RETRIEVER_SEARCH": "10",
            "K2_RETRIEVER_SEARCH": "20",
            "D_RETRIEVER_SEARCH": "invalid",
        }.get(x)
        with self.assertRaises(ValueError):
            load_hyperparams()
