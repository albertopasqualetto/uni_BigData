from pyspark import SparkContext, SparkConf
import sys

def main():
    conf = SparkConf().setAppName('G007HW1')
    sc = SparkContext(conf=conf)
    argc = len(sys.argv)
    if argc != 5:
        print('Usage: python G007HW1.py <file_name> <D> <M> <K> <L>')
        exit(1)
    file_name = sys.argv[1]
    D = int(sys.argv[2])
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

def ApproxOutliers(points, D):
    points = round1(points, D)
    #TODO: Implement the other rounds of the approximate algorithm
    return points

def round1(points, D):
    return points.flatmap(lambda x,y : ((x/D,y/D),1)).reduceByKey(lambda x,y : x+y)  
    

if __name__ == "__main__":
    main()
