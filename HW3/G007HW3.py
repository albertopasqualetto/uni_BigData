from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import numpy as np


# All 3 method: all items after the n-th one should be ignored

# After how many items should we stop?
n = -1  # To be set via command line
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
streamLength = [0]  # Stream length (an array to be passed by reference)
S_exact = {}
S = {}
S_reservoir = []
t = 0 # number of item processed
S_sticky = {}  # Hash Table


def exactStep(item):
    global S
    if item in S.keys():
        S[item] = S[item] + 1
    else:
        S[item] = 1


def reservoirSamplingStep(item):
    global phi, S_reservoir, t
    m = 1 / phi
    if t <= m:
        S_reservoir.append(item)
    else:
        x = np.random.uniform() # Random number in [0,1]
        p = np.ceil(m / t)
        if x <= p:
            S_reservoir.pop(np.random.randint(0, m))
            S_reservoir.append(item)


def stickySamplingStep(item, n, phi, epsilon, delta):
    global S_sticky
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
    global streamLength, S_exact
    global S_reservoir, S_sticky, t
    batch_size = batch.count()
    # If we already have enough points (> n), skip this batch.
    if streamLength[0] >= n:
        return

    batch = batch.map(lambda s: int(s))
    streamLength[0] += batch_size

    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    batch = batch.collect()
    # Update the streaming state
    for item in batch:
        t += 1
        if t > n:
            break
        exactStep(item)
        reservoirSamplingStep(item)
        stickySamplingStep(item, n, phi, epsilon, delta)

    if streamLength[0] >= n:
        stopping_condition.set()
        #print('received:',batch[:n])
        #print(t)

def compute_print_exact(S_exact):
    # The size of the data structure used to compute the true frequent items
    # The number of true frequent items
    # The true frequent items, in increasing order (one item per line)
    print("Data structure size:", len(S_exact)) # TODO understand what first point means
    print("Exact frequent items length:", len(S_exact))
    print("Exact frequent items:")
    for k in S_exact:
        print(k)


def compute_print_reservoir(S_reservoir, S_exact):
    # The size m of the Reservoir sample
    # The number of estimated frequent items (i.e., distinct items in the sample)
    # The estimated frequent items, in increasing order (one item per line). Next to each item print a "+" if the item is a true freuent one, and "-" otherwise.
    print("Reservoir sampling length:", len(S_reservoir))
    global sc
    S_reservoir = sc.parallelize(S_reservoir).map(lambda x: (x, 1)).reduceByKey(lambda i1, i2: i1 + i2).collectAsMap()  # distinct items
    print("Number of approximate frequent items:", len(S_reservoir))
    print("Reservoir sampling:")
    for k in sorted(S_reservoir.keys()):
        if k in S_exact:
            print(k, '+')
        else:
            print(k, '-')


def compute_print_sticky(S_sticky, S_exact):
    # The size of the Hash Table
    # The number of estimated frequent items (i.e., the items considered frequent by Sticky sampling)
    # The estimated frequent items, in increasing order (one item per line). Next to each item print a "+" if the item is a true freuent one, and "-" otherwise.
    print("Sticky sampling length:", len(S_sticky))
    S_sticky_frequent = {k: v for k, v in sorted(S_sticky.items()) if v >= (phi - epsilon) * n}
    print("Number of approximate frequent items:", len(S_sticky_frequent))
    print("Sticky sampling epsilon-approximate frequent items:")
    for k in S_sticky_frequent.keys():
        if k in S_exact:
            print(k, '+')
        else:
            print(k, '-')


def main():
    global S, n, phi, epsilon, delta, sc, stopping_condition, streamLength, S_exact, S_reservoir, t_reservoir, S_sticky, t_sticky
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
    for k in sorted(S.keys()):
        if S[k] >= n * phi:
            S_exact[k] = S[k]
    compute_print_exact(S_exact)
    compute_print_reservoir(S_reservoir, S_exact)
    compute_print_sticky(S_sticky, S_exact)

    # print("Number of items processed =", streamLength[0])
    # print("Number of distinct items =", len(S_exact))
    # print("Largest item =", max(S_exact.keys()))


if __name__ == '__main__':
    main()
