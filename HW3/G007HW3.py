from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import numpy as np
import random as rnd
from collections import defaultdict

# After how many items should we stop?
n = -1  # To be set via command line


def stickySamplingStep(batch_items, n, phi, epsilon, delta, S_sticky):
    local_batch_items = batch_items.collectAsMap()
    r = np.log(1/(delta*phi)) / (epsilon)
    p = r/n
    for key in local_batch_items:
        if key in S_sticky:
            S_sticky[key] = S_sticky[key] + 1
        else:
            x = rnd.uniform(0, 1) # Random number in [0,1]
            if x <= p:
                S_sticky[key] = 1


def reservoirSamplingStep(batch_items, phi, S_reservoir):
    local_batch_items = batch_items.collectAsMap()
    m = 1 / phi
    t = 0

    for key in local_batch_items:
        if t <= m:
            S_reservoir[key] = 1
        else:
            x = rnd.uniform(0, 1)  # Random number in [0,1]
            p = m / t
            if x <= p:
                S_reservoir[key] = 1


def exactStep(batch_items, S_exact):
    local_batch_items = batch_items.reduceByKey(lambda i1, i2: i1 + i2).collectAsMap()
    for key in local_batch_items:
        if key in S_exact:
            S_exact[key] = S_exact[key] + local_batch_items[key]
        else:
            S_exact[key] = local_batch_items[key]


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, S_exact
    global S_sticky, S_reservoir
    batch_size = batch.count()
    # If we are about to exceed the number of items we need to process, limit the batch to the max number n.
    if streamLength[0] >= n and streamLength[0] + batch_size <= n:
        diff = streamLength[0] + batch_size - n
        # batch = batch.toDF().limit(n).rdd
    # If we already have enough points (> n), skip this batch.
    elif streamLength[0] >= n:
        return

    streamLength[0] += batch_size
    # Extract items and frequency from the batch
    batch_items = batch.map(lambda s: (int(s), 1))

    # Update the streaming state
    exactStep(batch_items, S_exact)
    reservoirSamplingStep(batch_items, phi, S_reservoir)
    stickySamplingStep(batch_items, n, phi, epsilon, delta, S_sticky)

    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= n:
        stopping_condition.set()


def print_exact():
    global k
    S_exact_frequent = {k: v for k, v in sorted(S_exact.items()) if v >= n * phi}
    print("Exact frequent items length:", len(S_exact_frequent))
    print("Exact frequent items:")
    for k in S_exact_frequent.keys():
        print(k)


def print_reservoir():
    global k
    print("Reservoir sampling:")
    for k in S_reservoir.keys():
        print(k)


def extract_sticky():
    global k
    print("Sticky sampling length:", len(S_sticky))
    S_sticky_frequent = {k: v for k, v in sorted(S_sticky.items()) if v >= (phi - epsilon) * n}
    print("Sticky sampling epsilon-approximate frequent items:")
    for k in S_sticky_frequent.keys():
        print(k)


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc != 6:
        print('Usage: python3 G007HW3.py <n> <phi> <epsilon> <delta> <portExp>')
        sys.exit(1)

    n = int(sys.argv[1])            # An integer 𝑛: the number of items of the stream to be processed
    phi = float(sys.argv[2])        # A float phi: the frequency thresold in (0,1)
    epsilon = float(sys.argv[3])    # A float epsilon: the accuracy parameter in (0,1)
    delta = float(sys.argv[4])      # A float delta: the confidence parameter in (0,1)
    portExp = int(sys.argv[5])      # An integer portExp: the port number

    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("DistinctExample")
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

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    streamLength = [0]  # Stream length (an array to be passed by reference)
    S_exact = {}
    S_sticky = {}   # Hash Table
    S_reservoir = defaultdict(int)

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    # NOTE: You will see some data being processed even after the
    # shutdown command has been issued: This is because we are asking
    # to stop "gracefully", meaning that any outstanding work
    # will be done.
    ssc.stop(False, True)
    print("Streaming engine stopped")

    # COMPUTE AND PRINT FINAL STATISTICS
    print_exact()
    print_reservoir()
    extract_sticky()

    # print("Number of items processed =", streamLength[0])
    # print("Number of distinct items =", len(S_exact))
    # print("Largest item =", max(S_exact.keys()))
