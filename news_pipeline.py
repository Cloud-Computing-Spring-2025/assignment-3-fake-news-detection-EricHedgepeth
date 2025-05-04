from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import concat_ws
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("FakeNewsPipeline") \
    .getOrCreate()



news_df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)




news_df.createOrReplaceTempView("news_data")




# a) First 5 rows
spark.sql("SELECT * FROM news_data LIMIT 5").show()

# b) Total count
spark.sql("SELECT COUNT(*) AS total_articles FROM news_data").show()

# c) Distinct labels
spark.sql("SELECT DISTINCT label FROM news_data").show()



news_df.coalesce(1) \
    .write.option("header", True) \
    .csv("task1_output.csv")






lower_df = news_df.withColumn("text_lower", lower(col("text")))


tokenizer = Tokenizer(inputCol="text_lower", outputCol="words_raw")
tok_df = tokenizer.transform(lower_df)


remover = StopWordsRemover(inputCol="words_raw", outputCol="filtered_words")
cleaned_df = remover.transform(tok_df)



cleaned_df.createOrReplaceTempView("cleaned_news")


task2_df = cleaned_df.withColumn(
    "filtered_words",
    concat_ws(" ", col("filtered_words")))

task2_df.select("id","title","filtered_words","label") \
   .coalesce(1).write.option("header", True) \
   .csv("task2_output.csv")


hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
tf_df = hashingTF.transform(cleaned_df)


idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf_df)
tfidf_df = idfModel.transform(tf_df)


indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed_df = indexer.fit(tfidf_df).transform(tfidf_df)



task3_df = indexed_df \
    .withColumn("filtered_words", concat_ws(" ", col("filtered_words"))) \
    .withColumn("features", col("features").cast("string"))


task3_df.select("id", "filtered_words", "features", "label_index") \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", True) \
    .csv("task3_output.csv")


train_df, test_df = indexed_df.randomSplit([0.8,0.2], seed=42)


lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lrModel = lr.fit(train_df)


predictions = lrModel.transform(test_df)


predictions.select("id","title","label_index","prediction") \
    .coalesce(1).write.option("header",True) \
    .csv("task4_output.csv")


evaluator = MulticlassClassificationEvaluator(labelCol="label_index")



accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f"Accuracy = {accuracy:.4f}, F1 = {f1_score:.4f}")




metrics_df = spark.createDataFrame([
    ("Accuracy", accuracy),
    ("F1 Score", f1_score)
], ["Metric","Value"])

metrics_df.coalesce(1).write.option("header",True) \
    .csv("task5_output.csv")
