## Ratings Counter
1

	val sc = new SparkContext("local[*]", "RatingsCounter")

* Create a SparkContext object
* "local" means standalone mode
* " * " means using all the cores of the local machine

2

	val lines = sc.textFile("../ml-100k/u.data")

lines are all the rows of the dataset

	user-id  movie-id  rating  timestamp
	196      242       3       881250949
	186      302       3       891717742
	22       377       1       878887116
	244      51        2       880606923
	166      346       1       886397596

3

	val ratings = lines.map(x => x.toString().split("\t")(2))

* call map on the RDD lines to transfer to a new RDD called ratings
* take every individual line, transfer to string, then split and extract field number two from the list

**=>**

		3
		3
		1
		2
		1

4

	val results = ratings.countByValue()

  3 occurs 2 times, 1 occurs 2 times, 2 occurs 1 times 


	(3, 2)
	(1, 2)
	(2, 1)

5

	val sortedResults = results.toSeq.sortBy(_._1)<br> 
	sortedResults.foreach(println)

* Sort the resulting map of (rating, count) tuples
* `_._1` gets the first element of a tuple