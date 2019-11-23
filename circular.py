class MyCircularQueue:
    # tail always be the next insertion point, head always the head. circular just need to mod the size and
    # everything else would be the same as non-circular
    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.k = k
        self.size = 0
        self.q = [0 for _ in range(k)]
        self.head = self.tail = 0

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        self.q[self.tail] = value
        self.tail = (self.tail + 1) % self.k
        self.size += 1
        return True

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.k
        self.size -= 1
        return True

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        return self.q[self.head] if not self.isEmpty() else -1

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        return self.q[self.tail - 1] if not self.isEmpty() else -1

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.size == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.size == self.k