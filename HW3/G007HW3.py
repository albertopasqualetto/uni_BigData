from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import numpy as np

# What to ask monday:
# - Exception in thread "receiver-supervisor-future-0" java.lang.InterruptedException: sleep interrupted
	# at java.base/java.lang.Thread.sleep(Native Method)
	# at org.apache.spark.streaming.receiver.ReceiverSupervisor.$anonfun$restartReceiver$1(ReceiverSupervisor.scala:196)
	# at scala.runtime.java8.JFunction0$mcV$sp.apply(JFunction0$mcV$sp.java:23)
	# at scala.concurrent.Future$.$anonfun$apply$1(Future.scala:659)
	# at scala.util.Success.$anonfun$map$1(Try.scala:255)
	# at scala.util.Success.map(Try.scala:213)
	# at scala.concurrent.Future.$anonfun$map$1(Future.scala:292)
	# at scala.concurrent.impl.Promise.liftedTree1$1(Promise.scala:33)
	# at scala.concurrent.impl.Promise.$anonfun$transform$1(Promise.scala:33)
	# at scala.concurrent.impl.CallbackRunnable.run(Promise.scala:64)
	# at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1136)
	# at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:635)
	# at java.base/java.lang.Thread.run(Thread.java:842)
# - Approximate the len of the stream with n because we need it in the sticky sampling?
# - Sticky sampling: if we have already n items processed, we should stop the computation?

# After how many items should we stop?
n = -1  # To be set via command line
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
streamLength = [0]  # Stream length (an array to be passed by reference)
S_exact = {}
S_reservoir = []
t_reservoir = 1
S_sticky = {}  # Hash Table
t_sticky = 1  # number of items processed


def exactStep(batch_items):
    global S_exact
    local_batch_items = batch_items.map(lambda s: (s, 1)).reduceByKey(lambda i1, i2: i1 + i2).collectAsMap()
    for key in local_batch_items:
        if key in S_exact:
            S_exact[key] = S_exact[key] + local_batch_items[key]
        else:
            S_exact[key] = local_batch_items[key]


def reservoirSamplingStep(item):
    global phi, S_reservoir, t_reservoir
    m=1/phi
    #print("step:", t_reservoir, "reservoir:", S_reservoir)
    #print("step id:", id(S_reservoir))
    # print("step:", t_reservoir, "reservoir:", S_reservoir)
    if t_reservoir <= m:
        S_reservoir.append(item)
    else:
        x = np.random.uniform() # Random number in [0,1]
        p = np.ceil(m / t_reservoir)
        if x <= p:
            S_reservoir.pop(np.random.randint(0, m))
            S_reservoir.append(item)
    t_reservoir += 1


def stickySamplingStep(item, n, phi, epsilon, delta):
    global t_sticky, S_sticky
    #print('method:', S_sticky)
    if t_sticky > n:
        return
    r = np.log(1/(delta*phi)) / epsilon
    p = r/n

    if item in S_sticky:
        S_sticky[item] = S_sticky[item] + 1
    else:
        x = np.random.uniform() # Random number in [0,1]
        if x <= p:
            S_sticky[item] = 1
    t_sticky+=1


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, S_exact
    global S_reservoir, t_reservoir, S_sticky, t_sticky
    #print("batch", id(S_reservoir))
    batch_size = batch.count()
    # If we are about to exceed the number of items we need to process, limit the batch to the max number n.    # TODO use this limit for all algorithms or just for sticky as written down in the call?
    # if streamLength[0] >= n >= streamLength[0] + batch_size:
    #     diff = streamLength[0] + batch_size - n
    #     batch = batch.zipWithIndex().filter(lambda x: x[1] < diff).map(lambda x: x[0])  # this will trigger an action (strict less than since index starts from 0)
    # If we already have enough points (> n), skip this batch.
    if streamLength[0] >= n:
        return

    batch = batch.map(lambda s: int(s))

    streamLength[0] += batch_size

    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    # Update the streaming state
    exactStep(batch)
    for item in batch.collect():
        reservoirSamplingStep(item)
        stickySamplingStep(item, n, phi, epsilon, delta)
    # batch.foreach(reservoirSamplingStep)
    # print('main res', S_reservoir)
    # reservoirSamplingStep(batch_items, 1 / phi, t, S_reservoir)
    # batch.foreach(lambda item: stickySamplingStep(item, n, phi, epsilon, delta))
    # print('main',S_sticky)
    # stickySamplingStep(batch_items, n, phi, epsilon, delta, S_sticky)

    if streamLength[0] >= n:
        stopping_condition.set()


def compute_print_exact(S_exact):
    S_exact_frequent = {k: v for k, v in sorted(S_exact.items()) if v >= n * phi}
    print("Exact frequent items length:", len(S_exact_frequent))
    print("Exact frequent items:")
    for k in S_exact_frequent.keys():
        print(k)


def compute_print_reservoir(S_reservoir):
    print("Reservoir sampling length:", len(S_reservoir))
    global sc
    S_reservoir = sc.parallelize(S_reservoir).map(lambda x: (x, 1)).reduceByKey(lambda i1, i2: i1 + i2).collectAsMap()  # distinct items
    print("Reservoir sampling:")
    for k in sorted(S_reservoir.keys()):
        print(k)


def compute_print_sticky(S_sticky):
    print("Sticky sampling length:", len(S_sticky))
    S_sticky_frequent = {k: v for k, v in sorted(S_sticky.items()) if v >= (phi - epsilon) * n}
    print("Sticky sampling epsilon-approximate frequent items:")
    for k in S_sticky_frequent.keys():
        print(k)


def main():
    global n, phi, epsilon, delta, sc, stopping_condition, streamLength, S_exact, S_reservoir, t_reservoir, S_sticky, t_sticky
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
    compute_print_exact(S_exact)
    compute_print_reservoir(S_reservoir)
    compute_print_sticky(S_sticky)

    # print("Number of items processed =", streamLength[0])
    # print("Number of distinct items =", len(S_exact))
    # print("Largest item =", max(S_exact.keys()))


if __name__ == '__main__':
    main()
