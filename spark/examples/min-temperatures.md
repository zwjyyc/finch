## Min Temperatures
```scala
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.math.min

/** Find the minimum temperature by weather station */
object MinTemperatures {
  
  def parseLine(line:String)= {
    val fields = line.split(",")
    val stationID = fields(0)
    val entryType = fields(2)
    val temperature = fields(3).toFloat * 0.1f * (9.0f / 5.0f) + 32.0f
    (stationID, entryType, temperature)
  }
    /** Our main function where the action happens */
  def main(args: Array[String]) {
   
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Create a SparkContext using every core of the local machine
    val sc = new SparkContext("local[*]", "MinTemperatures")
    
    // Read each line of input data
    val lines = sc.textFile("../1800.csv")
    
    // Convert to (stationID, entryType, temperature) tuples
    val parsedLines = lines.map(parseLine)
    
    // Filter out all but TMIN entries
    val minTemps = parsedLines.filter(x => x._2 == "TMIN")
    
    // Convert to (stationID, temperature)
    val stationTemps = minTemps.map(x => (x._1, x._3.toFloat))
    
    // Reduce by stationID retaining the minimum temperature found
    val minTempsByStation = stationTemps.reduceByKey( (x,y) => min(x,y))
    
    // Collect, format, and print the results
    val results = minTempsByStation.collect()
    
    for (result <- results.sorted) {
       val station = result._1
       val temp = result._2
       val formattedTemp = f"$temp%.2f F"
       println(s"$station minimum temperature: $formattedTemp") 
    }
      
  }
}
```

#### 1.
Output is (stationID, entryType, temperature)
```scala
def parseLine(line:String)= {
	val fields = line.split(",")
	val stationID = fields(0)
	val entryType = fields(2)
	val temperature = fields(3).toFloat * 0.1f * (9.0f / 5.0f) + 32.0f
	(stationID, entryType, temperature)
}
// Read each line of input data
val lines = sc.textFile("../1800.csv")
// Convert to (stationID, entryType, temperature) tuples
val parsedLines = lines.map(parseLine)
```
#### 2.
Filter out (remove) entries that don't have "TMIN"
```scala
val minTemps = parsedLines.filter(x => x._2 == "TMIN")
```
#### 3.
Create (stationID, temperature) key / value pairs
```scala
val stationTemps = minTemps.map(x => (x._1, x._3.toFloat))
```
#### 4.
* Find minimum temperature by stationID
* For reduceByKey(), x is current value, y is new value
```
val minTempsByStation = stationTemps.reduceByKey( (x,y) => min(x,y) )
```
#### 5.
```scala
val results = minTempsByStation.collect()

for (result <- results.sorted) {
	val station = result._1
	val temp = result._2
	val formattedTemp = f"$temp%.2f F"
	println(s"$station minimum temperature: $formattedTemp") 
}
```
