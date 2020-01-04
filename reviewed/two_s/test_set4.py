from unittest import TestCase
from reviewed.two_s import MultiStack


class TestSet4(TestCase):
    def setUp(self) -> None:
        self.st = MultiStack()

    def test_st(self) -> None:
        self.st.push(3)
        self.st.push(5)
        self.assertEqual(self.st.get_min(), 3)
        self.assertEqual(self.st.get_max(), 5)
        self.assertEqual(self.st.get_avg(), 4.0)
        self.assertSetEqual(set(self.st.get_mode()), {3, 5})
        self.assertEqual(self.st.get_median(), 4.0)

        self.st.pop()
        self.st.push(7)
        self.assertEqual(self.st.get_min(), 3)
        self.assertEqual(self.st.get_max(), 7)
        self.assertSetEqual(set(self.st.get_mode()), {3, 7})
        self.assertEqual(self.st.get_median(), 5.0)

        self.st.push(3)
        self.assertSetEqual(set(self.st.get_mode()), {3})
        self.assertEqual(self.st.get_median(), 3)

        self.st.push(7)
        self.assertSetEqual(set(self.st.get_mode()), {3, 7})
        self.assertEqual(self.st.get_median(), 5.0)
        self.st.push(7)
        self.assertSetEqual(set(self.st.get_mode()), {7})
        self.assertEqual(self.st.get_median(), 7.0)
        self.st.pop()
        self.assertSetEqual(set(self.st.get_mode()), {3, 7})
        self.assertEqual(self.st.get_median(), 5.0)