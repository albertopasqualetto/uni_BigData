from pyspark import SparkContext, SparkConf
import sys
import time
import random as rnd

conf = SparkConf().setAppName('G007HW2')
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

C = None    # set of centers


def main():
    argc = len(sys.argv)
    if argc != 5:
        print('Usage: python3 G007HW2.py <file_name> <M> <K> <L>')
        sys.exit(1)
    file_name = sys.argv[1]
    M = int(sys.argv[2])    # threshold for outliers
    K = int(sys.argv[3])    # number of clusters
    L = int(sys.argv[4])    # number of partitions

    print(f"{file_name} M={M} K={K} L={L}")
    # import file into an RDD of strings (rawData)
    rawData = sc.textFile(file_name)

    # map rowData into a RDD of tuples of floats subdivided into L partitions (inputPoints)
    inputPoints = rawData\
                        .flatMap(lambda s: [tuple(float(x) for x in s.split(','))])\
                        .repartition(L)\
                        .cache()

    num = inputPoints.count()
    print("Number of points =", num)

    D = MRFFT(inputPoints, K)

    MRApproxOutliers(inputPoints, D, M)


def SequentialFFT(P, K):
    # P is the list of points
    # K is the number of clusters
    # returns a set C of K centers
    # O(|P|*K)
    S = []
    S.append(rnd.choice(P))
    d = {k: distance(k, S[0]) for k in P}
    for i in range(1, K):
        c = max(d, key=d.get)
        S.append(c)
        for p in P:
            d[p] = min(d[p], distance(p, c))
    return S


def MRFFT(P, K):
    # P is the list of points
    # K is the number of clusters
    # D is the radius (float)
    start_time_ns = time.time_ns()
    coreset = FFTround1(P, K).persist()
    coreset.count() # force the computation of the RDD
    end_time_ns = time.time_ns()
    print("Running time of MRFFT Round 1 =", (end_time_ns - start_time_ns) / (10 ** 6), "ms")
    #print("Coreset = ", coreset)
    start_time_ns = time.time_ns()
    centers = FFTround2(coreset.collect(), K)
    end_time_ns = time.time_ns()
    print("Running time of MRFFT Round 2 =", (end_time_ns - start_time_ns) / (10 ** 6), "ms")
    #print("Centers = ", centers)
    start_time_ns = time.time_ns()
    D = FFTround3(P).collect()[0][1]
    end_time_ns = time.time_ns()
    print("Running time of MRFFT Round 3 =", (end_time_ns - start_time_ns) / (10 ** 6), "ms")
    print("Radius =", D)
    return D


def FFTround1(P, K):
    # compute the coreset
    # map P into L subsets of equal size
    # reduce every subset with FFT
    return P\
            .mapPartitions(lambda p: SequentialFFT(list(p), K))


def FFTround2(coreset, K):
    # obtain the centers from SequentialFFT
    # empty map
    # compute the centers
    centers = SequentialFFT(coreset, K)

    global C
    C = sc.broadcast(centers)
    return centers


def FFTround3(points):
    # compute the radius R (float) of the clustering induced by the centers
    global C
    return points\
        .map(lambda pt: FFTmap_round3(pt, C))\
        .reduceByKey(lambda r1, r2: max(r1, r2))


def FFTmap_round3(point, C):
    # returns the distance between the point and the closest center "dist(x,C)"
    local_C = C.value
    nearest_center = min(local_C, key=lambda c: distance(point, c))
    return (0, distance(point, nearest_center)) # 0 is a dummy key to then group all the distances together


# returns the Euclidean distance between two points p1 and p2 expressed as tuples
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


# ApproxOutliers #######################################################################################################


def MRApproxOutliers(inputPoints, D, M):
    start_time_ns = time.time_ns()
    (points_per_cell, approx_out) = ApproxOutliersAlgo(inputPoints, M, D)
    end_time_ns = time.time_ns()
    results = {}
    for certainty, v in zip(approx_out[0:-1:2], approx_out[1::2]):
        results[certainty] = v
    print("Number of sure outliers =", results['outliers'][1] if 'outliers' in results.keys() else 0)
    print("Number of uncertain points =", results['uncertain'][1] if 'uncertain' in results.keys() else 0)
    #outputs = points_per_cell.collect()
    # for i in range(0, len(outputs)):
    #     print(f"Cell: ({outputs[i][0][0]},{outputs[i][0][1]})  Size = {outputs[i][1]}")
    print("Running time of MRApproxOutliers =", (end_time_ns - start_time_ns) / (10 ** 6), "ms")


def ApproxOutliersAlgo(points, M, D):
    points_per_cell = AOroundA(points, D).cache() # cache because after all the rounds we have to reuse this RDD
    points_square_3 = AOroundB_3(points_per_cell)
    points_square_7 = AOroundB_7(points_per_cell)

    # merge all the obtained results
    u = points_square_3.union(points_square_7).union(points_per_cell)
    u = u.groupByKey()\
                     .map(lambda x: (x[0], tuple(x[1])))

    outliers = AOroundC(u, M)
    return (points_per_cell, outliers.collect())


# from the set of points of the dataset, returns the cells of the grid and for each cell how many points are inside
def AOroundA(points, D):
    return points\
                .map(lambda pt: AOmap_roundA(pt, D))\
                .reduceByKey(lambda val1, val2: val1+val2)  # map a point into its cells and then counts the number of points inside each cell


# from the set of cells, returns the squares 3x3 with the number of points inside
def AOroundB_3(points_per_cell):
    return points_per_cell\
                        .flatMap(lambda cs: AOmap_roundB(cs, 3))\
                        .reduceByKey(AOreduce_roundB)


# from the set of cells, returns the squares 7x7 with the number of points inside
def AOroundB_7(points_per_cell):
    return points_per_cell\
                        .flatMap(lambda cs: AOmap_roundB(cs, 7))\
                        .reduceByKey(AOreduce_roundB)

# from the points per cell, points per 3x3 square and point per 7x7 square, 
# returns an RDD with "outliers", "non-outliers" and "uncertain" as key and with the list of cells of that type 
# and the number of points for each category
# 'empty' is used when the center cell of the square is empty
def AOroundC(cells, M):
    return cells\
                .map(lambda cs: AOmap_roundC(cs, M))\
                .groupByKey()\
                .flatMap(AOreduce_roundC)


# map each point into the cell it belongs, with value 1
def AOmap_roundA(point, D):
    # point = (x1, y1)
    SIDE = D/(2*(2**0.5))
    # if x or y is negative the starting index is -1, while 0 for the positive values
    if point[0] < 0:
        offsetNegX = -1
    else:
        offsetNegX = 0
    if point[1] < 0:
        offsetNegY = -1
    else:
        offsetNegY = 0
    return ((int((point[0]/SIDE)+offsetNegX), int((point[1]/SIDE)+offsetNegY)), 1)  # map each point in the cell it is in


# map each cell C into a list of squares such that they all contain the cell C
def AOmap_roundB(cell, square_dim):
    # cell = [(i1, j1), # of points in (i1, j1)]
    squares_cells = []
    for i in range(-int(square_dim/2), int(square_dim/2) + 1):
        for j in range(-int(square_dim/2), int(square_dim/2) + 1):
            if i == 0 and j == 0:  # if the current cell is the center
                squares_cells.append(((cell[0][0], cell[0][1]), (cell[1], 1)))      # ((i, j), (# of points in (i,j), 1))
            else:
                squares_cells.append(((cell[0][0]+i, cell[0][1]+j), (cell[1], 0)))  # ((i, j), (# of points in (i,j), 0))
    return squares_cells


# returns the number of points in the square, and 1 if the center cell contains points, 0 otherwise
def AOreduce_roundB(square1, square2):
    # square = (# of points in the square, 0 or 1)
    points_count = 0    # count of the points in the square
    center_count = 0    # count of valid centers (will result 1 only if the center cell contains points)
    points_count += (square1[0] + square2[0])
    center_count += (square1[1] + square2[1])
    return (points_count, center_count)


# map all the information into a pair with identifier ("outliers", "non-outliers", "uncertain", "empty") and the value is 
# the index of the cell that contain that type of points and the number of points it contains
def AOmap_roundC(cell, M):
    # cell = [(i, j), [(points_count3, center_count3), (points_count7, center_count7), # of points in (i,j)]]
    if cell[1][0][1] != 0:  # the zero means that the square is built around an empty cell
        N3 = cell[1][0][0]
        N7 = cell[1][1][0]
        if N3 > M:          # surely non-outliers
            return ('non outliers', (cell[0], cell[1][2]))
        elif N7 <= M:       # surely outliers
            return ('outliers', (cell[0], cell[1][2]))
        elif N3 <= M < N7:  # uncertain
            return ('uncertain', (cell[0], cell[1][2]))
    else:
        return ('empty', (cell[0], 0))  # the center cell of the square is empty


# returns an RDD with "outliers", "non-outliers", and "uncertain" as key and with the list of cells of that type 
# and the number of points for each category
# 'empty' is used for the center cell of the square that is empty
def AOreduce_roundC(cells):
    # cells = [outliers/non-outliers/uncertain/empty, [((i1, j1), # of points in (i1, j1)), ((i2, j2), # of points in (i2, j2)), ...]]
    list_square = []
    number_of_points = 0
    for cell in cells[1]:
        list_square.append(cell[0])
        number_of_points += cell[1]
    return (cells[0], (list_square, number_of_points))


if __name__ == "__main__":
    main()
