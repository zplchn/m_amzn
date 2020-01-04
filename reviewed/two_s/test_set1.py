from unittest import TestCase
from reviewed.two_s import Set1


class TestSet1(TestCase):
    def setUp(self) -> None:
        self.s = Set1()

    def test_empty_string(self):
        s, p = '', ''
        self.assertTrue(self.s.isMatch(s, p))

    def test_no_special_chars(self):
        s, p = 'abc', 'abc'
        self.assertTrue(self.s.isMatch(s, p))

        s, p = 'abc', 'abcd'
        self.assertFalse(self.s.isMatch(s, p))

        s, p = 'abcd', 'abc'
        self.assertFalse(self.s.isMatch(s, p))

    def test_star_is_zero_chars(self):
        s, p = 'abc', 'abc*'
        self.assertTrue(self.s.isMatch(s, p))

    def test_star_is_single_chars(self):
        s, p = 'abc', 'ab*'
        self.assertTrue(self.s.isMatch(s, p))

    def test_star_is_multi_chars(self):
        s, p = 'abc', 'a*'
        self.assertTrue(self.s.isMatch(s, p))

    def test_question_mark(self):
        s, p = 'abc', 'a?c'
        self.assertTrue(self.s.isMatch(s, p))


