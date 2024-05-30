from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import numpy as np
import random as rnd
# from collections import defaultdict # defaultdict(int)

# After how many items should we stop?
n = -1 # To be set via command line

def stickySampling(batch_items, n, phi, epsilon, delta):
    S = {} # Empty hash table
    r = np.log(1/(delta*phi)) / (epsilon)
    p = r/n
    for key in batch_items:
        if key in S:
            S[key] = S[key] + 1
        else:
            x = rnd.uniform(0, 1) # Random number in [0,1]
            if x <= p:
                S[key] = 1
    return S

def reservoirSampling(batch_items, phi):
    m = 1 / phi
    S = {}
    t = 0

    p = m / t
    for key in batch_items:
        if t <= m:
            S[key] = 1
        else:
            x = rnd.uniform(0, 1)  # Random number in [0,1]
            if x <= p:
                S[key] = 1


def bruteForce(batch_items, n, phi):
    for key, value in batch_items:
        if key in histogram.keys():
            histogram[key] = histogram[key] + value
        else:
            histogram[key] = value
    return histogram
    

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram
    batch_size = batch.count()
    # If we already have enough points (> n), skip this batch.
    if streamLength[0]>=n:
        return
    streamLength[0] += batch_size
    # Extract items and frequency from the batch
    batch_items = batch.map(lambda s: (int(s), 1)).reduceByKey(lambda i1, i2: i1+i2).collectAsMap()

    # Update the streaming state
    # for key in batch_items:
    #     if key not in histogram:
    #         histogram[key] = 1
    # call functions for the batch
    #stickySampling(batch_items, n, phi, epsilon, delta)
    #reservoirSampling()
    #histogram = bruteForce(batch_items, n, phi)
    # select only histogram[k]>=n*phi when printing
            
    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= n:
        stopping_condition.set()
        


if __name__ == '__main__':

    argc = len(sys.argv)
    if argc != 6:
        print('Usage: python3 G007HW3.py <n> <phi> <epsilon> <delta> <portExp>')
        sys.exit(1)

    n = sys.argv[1]               # An integer 𝑛: the number of items of the stream to be processed
    phi = int(sys.argv[2])        # A float phi: the frequency thresold in (0,1)
    epsilon = int(sys.argv[3])    # A float epsilon: the accuracy parameter in (0,1)
    delta = int(sys.argv[4])      # A float delta: the confidence parameter in (0,1)
    portExp = int(sys.argv[5])    # An integer portExp: the port number   

    # # print(f"n={n} phi={phi} epsilon={epsilon} delta={delta} portExp={portExp}")

    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("DistinctExample")
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    # conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    
    
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

    streamLength = [0] # Stream length (an array to be passed by reference)
    histogram = {} # Hash Table for the distinct elements
    

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
    print("Number of items processed =", streamLength[0])
    print("Number of distinct items =", len(histogram))
    largest_item = max(histogram.keys())
    print("Largest item =", largest_item)
    
