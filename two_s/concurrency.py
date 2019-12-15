import multiprocessing

def target_func(num):
    print(' This is %d of %s' % (num, multiprocessing.current_process().name))


processes = []
for i in range(5):
    p = multiprocessing.Process(target=target_func, args=(i,))
    processes.append(p)
    p.start()