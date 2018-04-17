import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Question1").master("local[*]").getOrCreate()

    val data = spark.read.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("src/data/adultTrain.csv")
      .na.drop()

    val featurizedData = FeatureExtraction.getFeaturizedDf(data)
    val Array(trainingData, testData) = featurizedData.randomSplit(Array(0.7, 0.3))
    val model = RandomForestClassifier.train(trainingData)

    val testPredictions = RandomForestClassifier.predict(model, testData)
    val trainingPredictions = RandomForestClassifier.predict(model, trainingData)

    println("Test predictions:")
    OutputEvaluation.evaluatePredictions(testPredictions)
    println("training predictions:")
    OutputEvaluation.evaluatePredictions(trainingPredictions)

    spark.stop()
  }


}
