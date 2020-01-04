from unittest import TestCase
from reviewed.two_s import LRUCache


class TestLRUCache(TestCase):
    def setUp(self) -> None:
        self.lru = LRUCache(10)

    def test_invalid_cap(self) -> None:
        with self.assertRaises(ValueError):
            LRUCache(-1)

    def test_empty(self) -> None:
        self.assertEqual(self.lru.get('a'), -1)

    def test_single_item(self) -> None:
        self.lru.put('a', 1)
        self.assertEqual(self.lru.get('a'), 1)

    def test_multi_item(self) -> None:
        self.lru.put('a', 1)
        self.lru.put('b', 2)
        self.assertEqual(self.lru.get('a'), 1)
        self.assertEqual(self.lru.get('b'), 2)

    def test_beyond_cap(self) -> None:
        for i in range(20):
            self.lru.put('a' + str(i), i)

        for j in range(10):
            self.assertEqual(self.lru.get('a' + str(j)), -1)
        for j in range(10, 20):
            self.assertEqual(self.lru.get('a' + str(j)), j)


