from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import numpy as np


# Global variables
sc: SparkContext = None
stopping_condition: threading.Event = None
n: int = -1
phi: float = -1
epsilon: float = -1
delta: float = -1
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
t = 0  # number of item processed
# stream_length = 0  # Stream length
S_exact = {}
S_reservoir = []
S_sticky = {}  # Hash Table


def exactStep(S_exact, item):
    # update the data structure responsible of computing the exact frequent items
    if item in S_exact.keys():
        S_exact[item] = S_exact[item] + 1
    else:
        S_exact[item] = 1


def reservoirSamplingStep(S_reservoir, item, t, phi):
    # update the data structure responsible of maintaining the m-sample through reservoir sampling
    m = np.ceil(1 / phi)
    if t <= m:
        S_reservoir.append(item)
    else:
        x = np.random.uniform() # Random number in [0,1]
        p = np.ceil(m / t)
        if x <= p:
            S_reservoir.pop(np.random.randint(0, m))
            S_reservoir.append(item)


def stickySamplingStep(S_sticky, item, n, phi, epsilon, delta):
    # update the data structure according to the sticky sampling
    r = np.log(1/(delta*phi)) / epsilon
    p = r/n
    if item in S_sticky:
        S_sticky[item] = S_sticky[item] + 1
    else:
        x = np.random.uniform() # Random number in [0,1]
        if x <= p:
            S_sticky[item] = 1


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global stopping_condition
    global S_exact, S_reservoir, S_sticky
    global t, n, phi, epsilon, delta
    batch_size = batch.count()
    # If we already have enough points (> n), skip this batch.
    if t > n:
        return

    batch = batch.map(lambda s: int(s))

    # if batch_size > 0:
    #     print("Batch size at time [{0}] is: {1}".format(time, batch_size))
    #     print("stream_length: {0}, t: {1}".format(stream_length, t))
    #     print("Hash tables sizes: EXACT={0}, RESERVOIR={1}, STICKY={2}".format(len(S_exact), len(S_reservoir), len(S_sticky)))

    batch = batch.collect()
    # Update the data structures
    for item in batch:
        t += 1
        if t > n:   # All 3 methods: all items after the n-th one should be ignored
            break
        exactStep(S_exact, item)
        reservoirSamplingStep(S_reservoir, item, t, phi)
        stickySamplingStep(S_sticky, item, n, phi, epsilon, delta)

    if t > n:
        stopping_condition.set()
        # print('received:',batch[:n])
        # print(t)


def threshold_S(S, n, phi):
    # compute the exact frequent item starting from the dictionary S
    S_thresholded = {}
    for k in S.keys():
        if S[k] >= n * phi:
            S_thresholded[k] = S[k]
    return S_thresholded


def compute_print_exact(S_exact_thresholded, S_exact_all):
    # The size of the data structure used to compute the true frequent items
    # The number of true frequent items
    # The true frequent items, in increasing order (one item per line)
    print("EXACT ALGORITHM")
    print("Number of items in the data structure =", len(S_exact_all))
    print("Number of true frequent items =", len(S_exact_thresholded))
    print("True frequent items:")
    for k in sorted(S_exact_thresholded.keys()):
        print(k)


def compute_print_reservoir(S_reservoir, S_exact):
    # The size m of the Reservoir sample
    # The number of estimated frequent items (i.e., distinct items in the sample)
    # The estimated frequent items, in increasing order (one item per line). Next to each item print a "+" if the item is a true frequent one, and "-" otherwise.
    global sc
    print("RESERVOIR SAMPLING")
    print("Size m of the sample =", len(S_reservoir))
    S_reservoir_summed = sc.parallelize(S_reservoir).map(lambda x: (x, 1)).reduceByKey(lambda i1, i2: i1 + i2).collectAsMap()  # distinct items
    print("Number of estimated frequent items =", len(S_reservoir_summed))
    print("Estimated frequent items:")
    for k in sorted(S_reservoir_summed.keys()):
        if k in S_exact:
            print(k, '+')
        else:
            print(k, '-')


def compute_print_sticky(S_sticky, S_exact):
    # The size of the Hash Table
    # The number of estimated frequent items (i.e., the items considered frequent by Sticky sampling)
    # The estimated frequent items, in increasing order (one item per line). Next to each item print a "+" if the item is a true frequent one, and "-" otherwise.
    print("STICKY SAMPLING")
    print("Number of items in the Hash Table =", len(S_sticky))
    S_sticky_frequent = {k: v for k, v in S_sticky.items() if v >= (phi - epsilon) * n}
    print("Number of estimated frequent items =", len(S_sticky_frequent))
    print("Estimated frequent items:")
    for k in sorted(S_sticky_frequent.keys()):
        if k in S_exact:
            print(k, '+')
        else:
            print(k, '-')


def main():
    global sc, stopping_condition
    global t, n, phi, epsilon, delta
    global S_exact, S_reservoir, S_sticky

    argc = len(sys.argv)
    if argc != 6:
        print('Usage: python3 G007HW3.py <n> <phi> <epsilon> <delta> <portExp>')
        sys.exit(1)
    n = int(sys.argv[1])  # An integer 𝑛: the number of items of the stream to be processed
    phi = float(sys.argv[2])  # A float phi: the frequency threshold in (0,1)
    epsilon = float(sys.argv[3])  # A float epsilon: the accuracy parameter in (0,1)
    delta = float(sys.argv[4])  # A float delta: the confidence parameter in (0,1)
    portExp = int(sys.argv[5])  # An integer portExp: the port number

    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName('G007HW3')
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    # Here, with the duration you can control how large to make your batches.
    # Beware that the data generator we are using is very fast, so the suggestion
    # is to use batches of less than a second, otherwise you might exhaust the memory.
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")
    # TECHNICAL DETAIL:
    # The streaming spark context and our code and the tasks that are spawned all
    # work concurrently. To ensure a clean shut down we use this semaphore.
    # The main thread will first acquire the only permit available and then try
    # to acquire another one right after spinning up the streaming computation.
    # The second tentative at acquiring the semaphore will make the main thread
    # wait on the call. Then, in the `foreachRDD` call, when the stopping condition
    # is met we release the semaphore, basically giving "green light" to the main
    # thread to shut down the computation.
    # We cannot call `ssc.stop()` directly in `foreachRDD` because it might lead
    # to deadlocks.
    stopping_condition = threading.Event()

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    # MANAGING STREAMING SPARK CONTEXT
    # print("Starting streaming engine")
    ssc.start()
    # print("Waiting for shutdown condition")
    stopping_condition.wait()
    # print("Stopping the streaming engine")
    # NOTE: You will see some data being processed even after the
    # shutdown command has been issued: This is because we are asking
    # to stop "gracefully", meaning that any outstanding work
    # will be done.
    ssc.stop(False, True)
    # print("Streaming engine stopped")

    # print("Final stream length: {0}, t: {1}, computed elements: {2}".format(stream_length, t, sum(S_exact.values())))

    print('INPUT PROPERTIES')
    print('n =', n, 'phi =', phi, 'epsilon =', epsilon, 'delta =', delta, 'port =', portExp)

    # COMPUTE AND PRINT FINAL STATISTICS
    S_exact_thresholded = threshold_S(S_exact, n, phi)
    compute_print_exact(S_exact_thresholded, S_exact)
    compute_print_reservoir(S_reservoir, S_exact_thresholded)
    compute_print_sticky(S_sticky, S_exact_thresholded)


if __name__ == '__main__':
    main()
