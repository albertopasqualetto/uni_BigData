from pyspark import SparkContext, SparkConf
import sys
from scipy.spatial import distance_matrix

conf = SparkConf().setAppName('G007HW1')
sc = SparkContext(conf=conf)
# sc.setLogLevel("DEBUG")


def main():
    argc = len(sys.argv)
    if argc != 6:
        print('Usage: python3 G007HW1.py <file_name> <D> <M> <K> <L>')
        exit(1)
    file_name = sys.argv[1]
    D = float(sys.argv[2])  # radius
    M = int(sys.argv[3])    # threshold for outliers
    K = int(sys.argv[4])
    L = int(sys.argv[5])
    points = sc.textFile(file_name)\
                                    .flatMap(lambda s: [tuple(float(x) for x in s.split(','))])\
                                    .repartition(L)\
                                    .cache()
    num = points.count()
    print(num)
    print(points.collect())
    if num < 200000:
        print('ExactOutliers')
        outliers = ExactOutliers(points, M, D)
        print('\tNumber of outliers: ', len(outliers))
        print('\tOutliers: ', [outliers[i] for i in range(0, min(K, len(outliers)))])
    print('ApproxOutliers')
    result = ApproxOutliers(points, M, D)
    print(result)
    plot_points(points.collect(), D)


def ExactOutliers(points, M, D):
    points_list = points.collect()
    # print(points_list)

    dist_mat = distance_matrix(points_list, points_list)
    # print(dist_mat)

    outliers = []   # list of indexes of outliers
    for i, point in enumerate(dist_mat):
        if len(point[point < D]) <= M:
            outliers.append(i)
    # print(len(outliers))
    # print(outliers)
    return outliers


def ApproxOutliers(points, M, D):
    points_per_cell = roundA(points, D)
    points_square_3 = roundB_3(points_per_cell)
    points_square_7 = roundB_7(points_per_cell)
    u = sc.union([points_square_3, points_square_7, points_per_cell])
    u = u.groupByKey()\
                    .map(lambda x: (x[0], list(x[1])))
    # print(u.collect())
    outliers = roundC(u, M)
    return outliers.collect()


def roundA(points, D):
    return points\
                .mapPartitions(lambda pts: map_roundA(pts, D))\
                .reduceByKey(lambda val1, val2: val1+val2)  # count the number of points in each cell


def roundB_3(points_per_cell):
    return points_per_cell\
                        .mapPartitions(lambda cs: cell_mapping(cs, 3))\
                        .reduceByKey(square_reduce)


def roundB_7(points_per_cell):
    return points_per_cell\
                        .mapPartitions(lambda cs: cell_mapping(cs, 7))\
                        .reduceByKey(square_reduce)


def roundC(cells, M):   # return for each cell the number of outliers, non-outliers, and uncertain points
    return cells\
                .mapPartitions(lambda cs: map_roundC(cs, M))\
                .groupByKey()\
                .flatMap(reduce_roundC)


def map_roundA(points, D):
    SIDE = D/(2*(2**0.5))
    # print("SIDE", SIDE)
    val = []
    for i, point in enumerate(points):
        # print(f"point {i} ({point[0]}, {point[1]}) in square:", (int(point[0]/SIDE),int(point[1]/SIDE)))
        val.append(((int(point[0]/SIDE),int(point[1]/SIDE)), 1))
    return val


def cell_mapping(cells, square_dim):
    squares_cells = []
    for cell in cells:
        # range: from -1 to 1 if square_dim = 3; from -3 to 3 if square_dim = 7
        for i in range(-int(square_dim/2), int(square_dim/2) + 1):
            for j in range(-int(square_dim/2), int(square_dim/2) + 1):
                if i == 0 and j == 0:  # if the current cell is the center
                    squares_cells.append(((cell[0][0], cell[0][1]), (cell[1], 1)))   # ((q_x, q_y), (|L_j|, 1))
                else:
                    squares_cells.append(((cell[0][0]+i, cell[0][1]+j), (cell[1], 0)))   # ((q_x, q_y), (|L_j|, 0))
    # print(squares_cells)
    return squares_cells


def square_reduce(square1, square2):
    # square = [key, (points_count, 0/1)] but without the key
    points_count = 0    # count of the points in the square
    center_count = 0
    points_count += (square1[0] + square2[0])   # update the points count
    center_count += (square1[1] + square2[1])   # if there is a center 1 is added
    return (points_count, center_count) # return the number of points in the square and the number of points in the square's center cell


def map_roundC(cells, M):
    # cell = [key,[(points_count3, center_count3), (points_count7, center_count7), points_count]]
    val = []
    for cell in cells:
        print("mapRoundC cell", cell, "CENTER" if cell[1][0][1] != 0 else " ")
        if cell[1][0][1] != 0:
            N3 = cell[1][0][0]
            N7 = cell[1][1][0]
            if N3 >= M: # surely non-outliers
                val.append(('non_outliers', (cell[0], cell[1][2])))
            elif N7 <= M:   # surely outliers
                val.append(("outliers", (cell[0], cell[1][2])))
            elif N3 <= M <= N7:    # uncertain
                val.append(("uncertain", (cell[0], cell[1][2])))
    return val


def reduce_roundC(cells):
    # cells = [0/1/2, [(key1,points1), (key2,points2), ...]]
    # cell = [0/1/2, (key, points_count)]
    list_square = []
    number_of_points = 0
    for cell in cells[1]:
        list_square.append(cell[0])
        number_of_points += cell[1]
    return (cells[0], (list_square, number_of_points))


def plot_points(points_list, D):
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    for p in points_list:
        ax.add_patch(plt.Circle(p, D, color='k', fill=False))
    x = [p[0] for p in points_list]
    y = [p[1] for p in points_list]
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
