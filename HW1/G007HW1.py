from pyspark import SparkContext, SparkConf
import sys
from scipy.spatial import distance_matrix

conf = SparkConf().setAppName('G007HW1')
sc = SparkContext(conf=conf)


def main():
    argc = len(sys.argv)
    if argc != 5:
        print('Usage: python G007HW1.py <file_name> <D> <M> <K> <L>')
        exit(1)
    file_name = sys.argv[1]
    D = float(sys.argv[2])
    M = int(sys.argv[3])
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
        ExactOutliers(points, D)
        print('ApproxOutliers')
        ApproxOutliers(points, D)
    else:
        print('ApproxOutliers')
        ApproxOutliers(points)
    pass


def ExactOutliers(points, M, D):
    points_list = points.collect()
    # print(points_list)

    dist_mat = distance_matrix(points_list, points_list)    # TODO maybe this should be implemented manually
    # print(dist_mat)

    outliers = []   # list of indexes of outliers
    for i, point in enumerate(dist_mat):
        if len(point[point < D]) <= M:
            outliers.append(i)
    # print(len(outliers))
    # print(outliers)
    return len(outliers)


def ApproxOutliers(points, D, M):
    points_per_cell = roundA(points, D)
    points_square_3 = roundB_3(points_per_cell)
    points_square_7 = roundB_7(points_per_cell)
    u = sc.union([points_per_cell, points_square_3, points_square_7])
    u = u.groupByKey()
    outliers = roundC(u, M).collect()

    return outliers


def roundA(points, D):
    return points.flatmap(lambda x, y: ((int(x/D), int(y/D)), 1)).reduceByKey(lambda x, y: x+y) # count the number of points in each cell


def roundB_3(points_per_cell):
    pass


def roundB_7(points_per_cell):
    pass


def roundC(cells, M):
    pass


def plot_points(points_list, D):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for p in points_list:
        ax.add_patch(plt.Circle(p, D, color='k', fill=False))
    x = [p[0] for p in points_list]
    y = [p[1] for p in points_list]
    plt.scatter(x, y)
    for i in range(len(points_list)):
        plt.annotate(str(i), (x[i], y[i]))
    plt.show()


if __name__ == "__main__":
    main()
