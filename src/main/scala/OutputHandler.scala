import java.io.{File, PrintWriter}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object OutputHandler {

  val outputPath = "output/"

  def writeScore(predictions: DataFrame, fileName: String): Unit ={
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    val writer = new PrintWriter(new File(outputPath + fileName + ".txt"))
    writer.println(s"Accuracy = $accuracy")
    writer.close()
  }

  def writeOutput(predictions: DataFrame, columnsToDrop: Array[String], fileName: String): Unit ={
    predictions.drop(columnsToDrop:_*).write
      .option("header", "true")
      .mode(SaveMode.Overwrite)
      .csv(outputPath + fileName)
  }
}
