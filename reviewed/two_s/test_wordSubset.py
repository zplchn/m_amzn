from unittest import TestCase
from reviewed.two_s import WordSubset


class TestWordSubset(TestCase):
    def setUp(self) -> None:
        self.ws = WordSubset()

    def test_find_substrings(self):
        word_dict = ['c', 'cd', 'ce', 'def', 'abcdefg']
        s = 'abcdefg'
        res = self.ws.find_substrings(s, word_dict)
        self.assertSetEqual(set(res), {'c', 'cd', 'def', 'abcdefg'})

