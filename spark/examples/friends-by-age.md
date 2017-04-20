## Friends By Age
1.
```scala
 def parseLine(line: String) = {
     // Split by commas
     val fields = line.split(",")
     // Extract the age and numFriends fields, and convert to integers
     val age = fields(2).toInt
     val numFriends = fields(3).toInt
     // Create a tuple that is our result.
     (age, numFriends)
 }
// Load each line of the source data into an RDD
val lines = sc.textFile("../fakefriends.csv")  
// Use our parseLines function to convert to (age, numFriends) tuples
val rdd = lines.map(parseLine)
```
A function has been defined to parse the csv below

	id      name      age    number of friends
	0       Will      33     385
	1       Jean-Luc  33     2
	2       Hugh      55     221
	3       Deanna    40     465
	4      	Quark     68     21

to

	Output is key/value pairs of (age, numFriends):	
	33, 385
	33, 2
	55, 221,
	40, 465

2.
```scala
val totalsByAge = rdd.mapValues(x => (x, 1)).reduceByKey( (x,y) => (x._1 + y._1, x._2 + y._2) )
```
This involves two steps:
```scala
rdd.mapValues(x=>(x, 1))
```
	(33, 385) => (33, (385, 1))
	(33, 2) => (33, (2, 1))
	(55, 221) => (55, (221, 1))
Adds up all values for each unique key
```scala
reduceByKey( (x,y) => (x._1 + y._1, x._2 + y._2) )
```
	  (33, (385, 1))
	+ (33, (2, 1)) 
	=> (33, (387, 2))

3.
```scala
val averagesByAge = totalsByAge.mapValues(x => x._1 / x._2)
```
	(33, (387, 2) => (33, 193.5)

4.
Collect the results from the RDD
```scala
val results = averagesByAge.collect()
```
Sort and print the final results
```scala
results.sorted.foreach(println)
```
