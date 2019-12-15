from unittest import TestCase
from two_s.set2 import Set2


class TestSet2(TestCase):
    def setUp(self) -> None:
        self.set2 = Set2()

    def test_power_of_4(self):
        x = [1, 2, 3, 4, 16, 4 ** 3, 100]
        r = [1, 0, 0, 1, 1, 1, 0]
        for i, j in zip(x, r):
            res = 1 if self.set2.power_of_4(i) else 0
            self.assertEqual(res, j)
