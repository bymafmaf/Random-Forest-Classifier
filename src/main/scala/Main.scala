import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Question1").master("local[*]").getOrCreate()

    val trainingData = spark.read.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("data/adultTrain.csv")
      .na.drop()

    val testData = spark.read.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("data/adultTest.csv")
      .na.drop()

    val categoricalFeatures = Array("workclass", "education", "marital_status", "occupation", "relationship", "race",
      "sex", "native_country")

    val trainingFeaturizedData = FeatureExtraction.getFeaturizedDf(trainingData, categoricalFeatures)
    val testFeaturizedData = FeatureExtraction.getFeaturizedDf(testData, categoricalFeatures)
    val model = RandomForestClassifier.train(trainingFeaturizedData)

    val testPredictions = RandomForestClassifier.predict(model, testFeaturizedData)
    val trainingPredictions = RandomForestClassifier.predict(model, trainingFeaturizedData)

    val columnsToDrop = categoricalFeatures.map(_ + "_indexed")++
      Array("features", "rawPrediction", "probability", "label", "prediction")

    OutputHandler.writeScore(testPredictions, "TestPredictionAccuracy")
    OutputHandler.writeOutput(testPredictions, columnsToDrop, "TestPrediction")

    OutputHandler.writeScore(trainingPredictions, "TrainingPredictionAccuracy")
    OutputHandler.writeOutput(trainingPredictions, columnsToDrop, "TrainingPrediction")

    spark.stop()
  }


}
