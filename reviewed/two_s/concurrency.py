import multiprocessing
from queue import Queue
from threading import Thread


def target_func(num):
    print(' This is %d of %s' % (num, multiprocessing.current_process().name))


processes = []
for i in range(5):
    p = multiprocessing.Process(target=target_func, args=(i,))
    processes.append(p)
    p.start()

# PCQueue is a thread-safe, unbounded queue.
# - push() appends an item to the end of the queue
# - poll() polls an item from the head of the queue. It can block waiting.
#
# Queue is not thread safe

# input source 1
class PCQueue:
    def push(self):
        pass
    def poll(self):
        pass

q1 = PCQueue()
# input source 2
q2 = PCQueue()
# we unify the two streams here
unified = PCQueue()
# helper queues
qs = [Queue(), Queue()]


def thread1():
    global q1, unified
    while True:
        head = q1.poll()
        unified.push((head, 1))


def thread2():
    global q2, unified
    while True:
        head = q2.poll()
        unified.push((head, 2))

def processing_thread():
    global unified
    while True:
        val, qid = unified.poll()

        myq = qs[qid]
        myq.append(val)
        # the other queue
        oq = qs[2 - qid]

        oq_new_begin = 0
        for i in range(len(oq)):
            oval = oq[i]
            diff = oval - val
            if diff < -1:
                oq_new_begin = i + 1
            elif diff > 1:
                break
            else:
                print(oval, val)
        del oq[:oq_new_begin]


t1 = Thread(target=thread1)
t2 = Thread(target=thread2)
t3 = Thread(target=processing_thread)

t1.start()
t2.start()
t3.start()