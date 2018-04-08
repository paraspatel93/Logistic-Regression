# Logistic-Regression
We will build a “mini Siri”, which will be capable of extracting flight information search terms from natural language using a logistic regression model.

We will implement a multinomial logistic regression model as the core of natural language processing
system.
Our goal in this assignment is to implement a working Natural Language Processing (NLP) system, i.e., a
mini Siri, using multinomial logistic regression. We will then use the algorithm to extract flight information
from natural text. We will do some very basic feature engineering, though which we will be able to improve
the learner’s performance on this task. We will write one program: tagger.py.

Eight command-line arguments:\<train input\> \<validation input\> \<test input\> \<train out\> \<test out\> \<metrics out\>
\<num epoch\> \<feature flag\>. These arguments are described in detail below:
1. \<train input\>: path to the training input .tsv file.
2. \<validation input\>: path to the validation input .tsv file.
3. \<test input\>: path to the test input .tsv file.
4. \<train out\>: path to output .labels file to which the prediction on the training data should be
written.
5. \<test out\>: path to output .labels file to which the prediction on the test data should be written.
6. \<metrics out\>: path of the output .txt file to which metrics such as train and test error should
be written.
7. \<num epoch>: integer specifying the number of times SGD loops through all of the training data
(e.g., if \<num epoch\> equals 5, then each training example will be used in SGD 5 times).
8. \<feature flag\>: integer taking value 1 or 2 that specifies whether to construct the Model 1
feature set or the Model 2 feature set—that is, if feature_flag==1 use Model
1 features; if feature_flag==2 use Model 2 features

As an example,the following command line would run your
program on the toy data provided for 2 epochs using the features from Model 1.

$ python tagger.py toytrain.tsv toyvalidation.tsv toytest.tsv \
mode1train_out.labels mode1test_out.labels mode1metrics_out.txt 2 1
