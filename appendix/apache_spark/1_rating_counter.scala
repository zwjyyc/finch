import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._

   
Logger.getLogger("org").setLevel(Level.FATAL)
    
val sc = new SparkContext("local[*]", "RatingsCounter")

val lines = sc.textFile("/Users/zhedongzheng/tutorials/apache_spark/temp/u.data")

val ratings = lines.map(x => x.toString().split("\t")(2))

val results = ratings.countByValue()

val sortedResults = results.toSeq.sortBy(_._1)

sortedResults.foreach(println)