sbt package
mkdir -p output
spark-submit \
  --class Main \
  --master local[*] \
  target/scala-2.11/organonquestion2_2.11-0.1.jar
