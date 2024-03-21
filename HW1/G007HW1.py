from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName('G007HW1')
    sc = SparkContext(conf=conf)

    docs = sc.textFile('TestN15-input.txt')\
                                    .flatMap(lambda s: [tuple(float(x) for x in s.split(','))])\
                                    .cache()
    num = docs.count()
    print()
    print(docs.collect())
    if num < 200000:
        print('ExactOutliers')
        ExactOutliers()
        print('ApproxOutliers')
        ApproxOutliers()
    else:
        print('ApproxOutliers')
        ApproxOutliers()
    pass

def ExactOutliers():
    pass

def ApproxOutliers():
    pass

if __name__ == "__main__":
    main()
