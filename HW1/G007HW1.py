from pyspark import SparkContext, SparkConf
import sys
from scipy.spatial import distance_matrix

conf = SparkConf().setAppName('G007HW1')
sc = SparkContext(conf=conf)


def main():
    argc = len(sys.argv)
    if argc != 6:
        print('Usage: python G007HW1.py <file_name> <D> <M> <K> <L>')
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
        ExactOutliers(points, D)
    print('ApproxOutliers')
    ApproxOutliers(points, D)



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
    return points.flatmap(lambda x: ((int(x[0]/D), int(x[1]/D)), 1)).reduceByKey(lambda val1, val2: val1+val2) # count the number of points in each cell
    
def cell_mapping(cell, square_dim): 
    squares = []
    #range: from -1 to 1 if square_dim = 3, from  -3 to 3 if square_dim = 7
    for i in range(-int(square_dim/2), int(square_dim/2)): 
        for j in range (-int(square_dim/2), int(square_dim/2)):
            if(i == 0 and j == 0): #if the current cell is the center
                squares.append((cell[0] + i, cell[1] + j), (cell[2], 1)) #((q_x, q_y), (|L_j|, 1))
            else:
                squares.append((cell[0], cell[1]), (cell[2], 0)) #((q_x, q_y), (|L_j|, 0))
    return squares

def square_reduce(square):
    center_count = 0
    points_count = 0 #count of the points in the square
    for cell in square:    
        points_count += cell[2] #update the points count
        center_count += cell[3] #if there is a center 1 is added 
    if center_count != 0: #if the center cell has at least one point
        return points_count

def roundB_3(points_per_cell):
    return points_per_cell.flatmap(lambda cell: cell_mapping(cell, 3)).reduceByKey(square_reduce)

def roundB_7(points_per_cell):
    return points_per_cell.flatmap(lambda cell: cell_mapping(cell, 7)).reduceByKey(square_reduce)


def roundC(cells, M): #return for each cell the number of outliers, non-outliers, and uncertain points
    return cells.flatMap(lambda cell : mapRoundC(cell, M)).groupByKey().mapValues(lambda vals: sum(vals))

def mapRoundC(cell, M):
    N3 = cell[2]
    N7 = cell[3]
    if N3 >= M: #surely non-outliers
        return [(2, cell[4])]
    elif N7 <= M: #surely outliers
        return [(0, cell[4])]
    elif N3 <=M and N7 >= M: #uncertain
        return [(1, cell[4])]


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
