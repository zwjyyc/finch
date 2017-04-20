## Min Temperatures
1.
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
2.
Filter out (remove) entries that don't have "TMIN"
```scala
val minTemps = parsedLines.filter(x => x._2 == "TMIN")
```
3.
Create (stationID, temperature) key / value pairs
```scala
val stationTemps = minTemps.map(x => (x._1, x._3.toFloat))
```
4.
* Find minimum temperature by stationID
* For reduceByKey(), x is current value, y is new value
```
val minTempsByStation = stationTemps.reduceByKey( (x,y) => min(x,y) )
```
5.
```scala
val results = minTempsByStation.collect()

for (result <- results.sorted) {
val station = result._1
val temp = result._2
val formattedTemp = f"$temp%.2f F"
println(s"$station minimum temperature: $formattedTemp") 
}
```
