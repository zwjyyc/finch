import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._

   
Logger.getLogger("org").setLevel(Level.ERROR)

val sc = new SparkContext("local[*]", "WordCountBetter")   

val input = sc.textFile("/Users/zhedongzheng/tutorials/apache_spark/temp/Book.txt")

val words = input.flatMap(x => x.split("\\W+"))

val lowercaseWords = words.map(x => x.toLowerCase())

val wordCounts = lowercaseWords.countByValue()

wordCounts.foreach(println)