from threading import Lock, Thread, Semaphore
from typing import Callable


class Foo1114:
    #pairwise synchronization
    def __init__(self):
        self.locks = [Lock(), Lock()]
        for l in self.locks:
            l.acquire()

    def first(self, printFirst: 'Callable[[], None]') -> None:
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.locks[0].release()

    def second(self, printSecond: 'Callable[[], None]') -> None:
        # printSecond() outputs "second". Do not change or remove this line.
        # wait for first job done lock to be released. and release the same lock after this context manager exist
        with self.locks[0]:
            printSecond()
            self.locks[1].release()

    def third(self, printThird: 'Callable[[], None]') -> None:
        # printThird() outputs "third". Do not change or remove this line.
        with self.locks[1]:
            printThird()


class FooBar1115:
    # everytime one block on its own lock and after done release the lock for the other. achieve interleaving
    def __init__(self, n):
        self.n = n
        self.foo_lock, self.bar_lock = Lock(), Lock()
        self.bar_lock.acquire()

    def foo(self, printFoo: 'Callable[[], None]') -> None:

        for i in range(self.n):
            # printFoo() outputs "foo". Do not change or remove this line.
            self.foo_lock.acquire()
            printFoo()
            self.bar_lock.release()

    def bar(self, printBar: 'Callable[[], None]') -> None:

        for i in range(self.n):
            # printBar() outputs "bar". Do not change or remove this line.
            self.bar_lock.acquire()
            printBar()
            self.foo_lock.release()


class ZeroEvenOdd1116:
    # three locks. 1 and 2 lock on itself and every time done release 0. 0 release alternatively between 1 and 2.
    def __init__(self, n):
        self.n = n
        self.locks = [Lock(), Lock(), Lock()] # lock for even, odd and 0
        self.locks[0].acquire()
        self.locks[1].acquire()
        self.cnt = 0

    # printNumber(x) outputs "x", where x is an integer.
    def zero(self, printNumber: 'Callable[[int], None]') -> None:
        for _ in range(self.n):
            self.locks[2].acquire()
            printNumber(0)
            self.cnt += 1
            self.locks[self.cnt % 2].release()

    def even(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(2, self.n + 1, 2):
            self.locks[0].acquire()
            printNumber(i)
            self.locks[2].release()

    def odd(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1, 2):
            self.locks[1].acquire()
            printNumber(i)
            self.locks[2].release()


class FizzBuzz1195:
    def __init__(self, n: int):
        self.n = n
        self.nn, self.f, self.b, self.fb = Semaphore(), Semaphore(0), Semaphore(0), Semaphore(0)
        self.cnt = 1

    # printFizz() outputs "fizz"
    def fizz(self, printFizz: 'Callable[[], None]') -> None:
        for i in range(3, self.n + 1, 3):
            if i % 15:
                self.f.acquire()
                printFizz()
                self.cnt += 1
                if self.cnt % 5 == 0:
                    self.b.release()
                else:
                    self.nn.release()

    # printBuzz() outputs "buzz"
    def buzz(self, printBuzz: 'Callable[[], None]') -> None:
        # for i in range(5, self.n + 1, 5): This causes when i == 15. no one release b. and this thread will starving
        for i in range(5, self.n + 1, 5):
            if i % 15:
                self.b.acquire()
                printBuzz()
                self.cnt += 1
                if self.cnt % 3 == 0:
                    self.f.release()
                else:
                    self.nn.release()


                    # printFizzBuzz() outputs "fizzbuzz"
    def fizzbuzz(self, printFizzBuzz: 'Callable[[], None]') -> None:
        for i in range(15, self.n + 1, 15):
            self.fb.acquire()
            printFizzBuzz()
            self.cnt += 1
            self.nn.release()

    # printNumber(x) outputs "x", where x is an integer.
    def number(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1, self.n + 1):
            if i % 3 and i % 5:
                self.nn.acquire()
                printNumber(i)
                self.cnt += 1
                if self.cnt % 3 == 0 and self.cnt % 5 == 0:
                    self.fb.release()
                elif self.cnt % 3 == 0:
                    self.f.release()
                elif self.cnt % 5 == 0:
                    self.b.release()
                else:
                    self.nn.release()


def print_number(x: int) -> None:
    print(x)


def main():
    threads = []
    # ze = ZeroEvenOdd1116(5)
    # for func in [ze.odd, ze.even, ze.zero]:
    #     threads.append(Thread(target=func, args=(print_number,)))
    #     threads[-1].start()
    # for t in threads:
    #     t.join()

    fb = FizzBuzz1195()
    for func in [fb.number, fb.f]:
        print('all done')


