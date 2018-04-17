import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.DataFrame

object RandomForestClassifier {

  def train(trainingData: DataFrame): PipelineModel ={
    val labelIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")
      .fit(trainingData)

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)
      .setMaxBins(41)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val mainPipeline = new Pipeline()
      .setStages(Array(
        labelIndexer,
        rf,
        labelConverter
      ))

    mainPipeline.fit(trainingData)
  }

  def predict(model: PipelineModel, testData: DataFrame): DataFrame ={
    model.transform(testData)
  }

}
