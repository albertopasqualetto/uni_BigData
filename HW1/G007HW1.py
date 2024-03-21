from pyspark import SparkContext, SparkConf
import sys

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


def ExactOutliers(points, D):
    pointsList = points.collect()
    # TODO: Implement the exact algorithm for finding outliers
    return pointsList


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


def roundC(cells, M):
    pass


if __name__ == "__main__":
    main()
