import unittest

from retrievers.process_dates import process_dates


class TestProcessDates(unittest.TestCase):
    def test_process_dates_with_valid_input(self):
        # Test with a valid input
        input_dates = ["2023-01-01", "2023-01-03", "2023-01-05"]
        d = 2
        expected_output = [
            "2022-12-30",
            "2022-12-31",
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
            "2023-01-07",
        ]
        self.assertEqual(process_dates(input_dates, d), expected_output)

    def test_process_dates_with_empty_input(self):
        # Test with an empty input
        input_dates = []
        d = 2
        expected_output = []
        self.assertEqual(process_dates(input_dates, d), expected_output)

    def test_process_dates_with_single_date(self):
        # Test with a single date in the input
        input_dates = ["2023-01-01"]
        d = 2
        expected_output = [
            "2022-12-30",
            "2022-12-31",
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
        ]
        self.assertEqual(process_dates(input_dates, d), expected_output)
