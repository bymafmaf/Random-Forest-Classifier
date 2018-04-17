import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

object FeatureExtraction {

  def getFeaturizedDf(data: DataFrame, categoricalFeatures: Array[String]): DataFrame ={
    val allNumericalData = handleCategoricalFeatures(data, categoricalFeatures)

    val featureList = categoricalFeatures.map(_ + "_indexed") ++
      Array("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")

    vectorizeFeatures(allNumericalData, featureList)
  }

  def handleCategoricalFeatures(data: DataFrame, categoricalFeatureList: Array[String]): DataFrame ={
    val categoricalIndexers = categoricalFeatureList.map{ feature =>
      new StringIndexer().setInputCol(feature).setOutputCol(feature + "_indexed")
    }
    val pipeline = new Pipeline().setStages(categoricalIndexers)

    pipeline.fit(data).transform(data)
  }

  def vectorizeFeatures(data: DataFrame, featureList: Array[String]): DataFrame ={
    val assembler = new VectorAssembler()
      .setInputCols(featureList)
      .setOutputCol("features")

    assembler.transform(data)
  }

}
