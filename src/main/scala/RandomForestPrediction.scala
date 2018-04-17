import org.apache.spark.sql.SparkSession

object RandomForestPrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Question1").master("local[*]").getOrCreate()

    val data = spark.read.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("src/data/adultTrain.csv")
      .na.drop()

    val categoricalFeatures = Array("workclass", "education", "marital_status", "occupation", "relationship", "race",
      "sex", "native_country")

    val featurizedData = FeatureExtraction.getFeaturizedDf(data, categoricalFeatures)
    val Array(trainingData, testData) = featurizedData.randomSplit(Array(0.7, 0.3))
    val model = RandomForestClassifier.train(trainingData)

    val testPredictions = RandomForestClassifier.predict(model, testData)
    val trainingPredictions = RandomForestClassifier.predict(model, trainingData)

    val columnsToDrop = categoricalFeatures.map(_ + "_indexed") ++ Array("features", "rawPrediction", "probability")

    OutputHandler.writeScore(testPredictions, "TestPredictionAccuracy")
    OutputHandler.writeOutput(testPredictions, columnsToDrop, "TestPrediction")

    OutputHandler.writeScore(trainingPredictions, "TrainingPredictionAccuracy")
    OutputHandler.writeOutput(trainingPredictions, columnsToDrop, "TrainingPrediction")

    spark.stop()
  }


}
