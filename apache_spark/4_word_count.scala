import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._

   
// Set the log level to only print errors
Logger.getLogger("org").setLevel(Level.ERROR)

  // Create a SparkContext using every core of the local machine
val sc = new SparkContext("local[*]", "WordCount")   

// Read each line of my book into an RDD
val lines = sc.textFile("/Users/zhedongzheng/tutorials/apache_spark/temp/Book.txt")

// Split into words separated by a space character
val words = lines.flatMap(x => x.split(" "))

// Count up the occurrences of each word
val wordCounts = words.countByValue()

// Print the results.
wordCounts.foreach(println)