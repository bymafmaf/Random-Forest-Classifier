# Random Forest Classifier with Apache Spark

This is a binary classifier model that utilizes Random Forest algorithm which is working with 10 decision trees to predict value of `class`

Data is randomly split into %70 and %30 pieces. It gives prediction with accuracy of `0.8411022576361222` for the test group.

Data source is https://archive.ics.uci.edu/ml/datasets/adult
Note: I had to change all "50K." values with "50K" in the adultTest dataset to make both dataset to have same values as label.

Required applications:
* Java JDK 1.8
* Scala 2.11
* SBT 1.x
* Apache Spark 2.x

## How to run?

Assuming all the applications are installed on the machine, you can just run
`./run-production.sh`

The `run-production.sh` script assumes that `spark-submit` binary can be called directly from the terminal.

## Output

You can find the output in `output` folder.
