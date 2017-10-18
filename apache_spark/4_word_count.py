from pyspark import SparkConf, SparkContext


conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)
sc.setLogLevel('FATAL')


lines = sc.textFile("/Users/zhedongzheng/tutorials/apache_spark/temp/Book.txt")
words = lines.flatMap(lambda x: x.split())
wordCounts = words.countByValue()


for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
