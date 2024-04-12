from pyspark import SparkContext, SparkConf
import sys
import time
import math


def main():
    argc = len(sys.argv)
    if argc != 6:
        print('Usage: python3 G007HW1.py <file_name> <D> <M> <K> <L>')
        sys.exit(1)
    file_name = sys.argv[1]
    D = float(sys.argv[2])  # exact algorithm radius: D, approximate algorithm cell diagonal: D/2
    M = int(sys.argv[3])    # threshold for outliers
    K = int(sys.argv[4])    # number of outliers/cells that will be printed
    L = int(sys.argv[5])    # number of partitions

    print(f"{file_name} D={D} M={M} K={K} L={L}")

    conf = SparkConf().setAppName('G007HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # import file into an RDD of strings (rawData)
    rawData = sc.textFile(file_name)

    # map rowData into a RDD of tuples of floats subdivided into L partitions (inputPoints)
    inputPoints = rawData\
                        .flatMap(lambda s: [tuple(float(x) for x in s.split(','))])\
                        .repartition(L)\
                        .cache()

    num = inputPoints.count()
    print("Number of inputPoints =", num)

    # EXACT ALGORITHM
    if num < 200000:
        listOfPoints = inputPoints.collect()
        ExactOutliers(listOfPoints, D, M, K)

    # APPROXIMATE ALGORITHM
    MRApproxOutliers(inputPoints, D, M, K)

    # plot_points(inputPoints.collect(), D)


def ExactOutliers(listOfPoints, D, M, K):
    start_time_ns = time.time_ns()
    outliers = ExactOutliersAlgo(listOfPoints, M, D)
    end_time_ns = time.time_ns()
    print('Number of Outliers = ', len(outliers))
    for i in range(0, min(K, len(outliers))):
        print(f"Point: ({outliers[i][0]},{outliers[i][1]})")
    print("Running time of ExactOutliers =", (end_time_ns - start_time_ns) / (10 ** 6), "ms")


def MRApproxOutliers(inputPoints, D, M, K):
    start_time_ns = time.time_ns()
    (points_per_cell, approx_out) = ApproxOutliersAlgo(inputPoints, M, D)
    end_time_ns = time.time_ns()
    results = {}
    for certainty, v in zip(approx_out[0:-1:2], approx_out[1::2]):
        results[certainty] = v
    print("Number of sure outliers =", results['outliers'][1] if 'outliers' in results.keys() else 0)
    print("Number of uncertain points =", results['uncertain'][1] if 'uncertain' in results.keys() else 0)
    first_K_nonempty = points_per_cell.takeOrdered(K, lambda x: x[1])
    for i in range(0, len(first_K_nonempty)):
        print(f"Cell: ({first_K_nonempty[i][0][0]},{first_K_nonempty[i][0][1]})  Size = {first_K_nonempty[i][1]}")
    print("Running time of ApproxOutliers =", (end_time_ns - start_time_ns) / (10 ** 6), "ms")


def ExactOutliersAlgo(points_list, M, D):
    outliers = {}
    # for each point the distance with all the points (itself and the others) is computed
    for i, current_point in enumerate(points_list):
        points_in_ball = 0
        for point in points_list:
            if math.dist(current_point, point) < D: # the point is inside the ball with radius D
                points_in_ball += 1
        if points_in_ball <= M: # the current point is an outlier
            outliers[i] = points_in_ball    # save the number of points in the ball using the index of the point as key
    return tuple(points_list[k] for k, v in sorted(outliers.items(), key=lambda item: item[1]))  # returns the list of outliers sorted by the number of points in the ball


def ApproxOutliersAlgo(points, M, D):
    points_per_cell = roundA(points, D).cache() # cache because after all the rounds we have to reuse this RDD
    points_square_3 = roundB_3(points_per_cell)
    points_square_7 = roundB_7(points_per_cell)

    # merge all the obtained results
    u = points_square_3.union(points_square_7).union(points_per_cell)
    u = u.groupByKey()\
                     .map(lambda x: (x[0], tuple(x[1])))

    outliers = roundC(u, M)
    return (points_per_cell, outliers.collect())


# from the set of points of the dataset, returns the cells of the grid and for each cell how many points are inside
def roundA(points, D):
    return points\
                .map(lambda pt: map_roundA(pt, D))\
                .reduceByKey(lambda val1, val2: val1+val2)  # map a point into its cells and then counts the number of points inside each cell


# from the set of cells, returns the squares 3x3 with the number of points inside
def roundB_3(points_per_cell):
    return points_per_cell\
                        .flatMap(lambda cs: map_roundB(cs, 3))\
                        .reduceByKey(reduce_roundB)


# from the set of cells, returns the squares 7x7 with the number of points inside
def roundB_7(points_per_cell):
    return points_per_cell\
                        .flatMap(lambda cs: map_roundB(cs, 7))\
                        .reduceByKey(reduce_roundB)


# from the points per cell, points per 3x3 square and point per 7x7 square, 
# returns an RDD with "outliers", "non-outliers" and "uncertain" as key and with the list of cells of that type 
# and the number of points for each category
# 'empty' is used for the center cell of the square that is empty
def roundC(cells, M):
    return cells\
                .map(lambda cs: map_roundC(cs, M))\
                .groupByKey()\
                .flatMap(reduce_roundC)


# map each point into the cell it belongs, with value 1
def map_roundA(point, D):
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
def map_roundB(cell, square_dim):
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
def reduce_roundB(square1, square2):
    # square = (# of points in the square, 0 or 1)
    points_count = 0    # count of the points in the square
    center_count = 0    # count of valid centers (will result 1 only if the center cell contains points)
    points_count += (square1[0] + square2[0])
    center_count += (square1[1] + square2[1])
    return (points_count, center_count)


# map all the information into a pair with identifier ("outliers", "non-outliers", "uncertain", "empty") and the value is 
# the index of the cell that contain that type of points and the number of points it contains
def map_roundC(cell, M):
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
def reduce_roundC(cells):
    # cells = [outliers/non-outliers/uncertain/empty, [((i1, j1), # of points in (i1, j1)), ((i2, j2), # of points in (i2, j2)), ...]]
    list_square = []
    number_of_points = 0
    for cell in cells[1]:
        list_square.append(cell[0])
        number_of_points += cell[1]
    return (cells[0], (list_square, number_of_points))


# (not requested, used for debugging) plot the points, the grid with squares of diagonal D/2 and the ball of each point
def plot_points(points_list, D):
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    for p in points_list:
        ax.add_patch(plt.Circle(p, D, color='k', fill=False))
    x = tuple(p[0] for p in points_list)
    y = tuple(p[1] for p in points_list)
    plt.scatter(x, y)
    for i in range(len(points_list)):
        plt.annotate(str(i), (x[i], y[i]))
    SIDE = D/(2*(2**0.5))
    ax.set_xticks(np.arange(0, max(x)+SIDE, SIDE))
    ax.set_yticks(np.arange(0, max(y)+SIDE, SIDE))
    ax.grid(which='both')
    plt.show()


if __name__ == "__main__":
    main()
