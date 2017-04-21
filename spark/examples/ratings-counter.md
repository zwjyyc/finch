## Ratings Counter
```scala
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._

/** Count up how many of each star rating exists in the MovieLens 100K data set. */
object RatingsCounter {
 
  /** Our main function where the action happens */
  def main(args: Array[String]) {
   
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
        
    // Create a SparkContext using every core of the local machine, named RatingsCounter
    val sc = new SparkContext("local[*]", "RatingsCounter")
   
    // Load up each line of the ratings data into an RDD
    val lines = sc.textFile("../ml-100k/u.data")
    
    // Convert each line to a string, split it out by tabs, and extract the third field.
    // (The file format is userID, movieID, rating, timestamp)
    val ratings = lines.map(x => x.toString().split("\t")(2))
    
    // Count up how many times each value (rating) occurs
    val results = ratings.countByValue()
    
    // Sort the resulting map of (rating, count) tuples
    val sortedResults = results.toSeq.sortBy(_._1)
    
    // Print each result on its own line.
    sortedResults.foreach(println)
  }
}
```
#### 1
* Create a SparkContext object
* "local" means standalone mode
* " * " means using all the cores of the local machine
```scala
val sc = new SparkContext("local[*]", "RatingsCounter")
```
#### 2

	user-id  movie-id  rating  timestamp
	196      242       3       881250949
	186      302       3       891717742
	22       377       1       878887116
	244      51        2       880606923
	166      346       1       886397596
lines are all the rows of the dataset
```scala
val lines = sc.textFile("../ml-100k/u.data")
```
#### 3
* call map on the RDD lines to transfer to a new RDD called ratings
* take every individual line, transfer to string, then split and extract field number two from the list
```scala
val ratings = lines.map(x => x.toString().split("\t")(2))
```

	3
	3
	1
	2
	1

#### 4
```scala
val results = ratings.countByValue()
```
3 occurs 2 times, 1 occurs 2 times, 2 occurs 1 times

	(3, 2)
	(1, 2)
	(2, 1)
#### 5
* Sort the resulting map of (rating, count) tuples
* `_._1` gets the first element of a tuple
```scala
val sortedResults = results.toSeq.sortBy(_._1)
sortedResults.foreach(println)
```

