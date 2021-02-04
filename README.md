# k-Nearest-Neighbors-and-k-fold-Cross-Validation

## k-Nearest Neighbors: ##

In this method the k most similar records from the train dataset to the record from the test dataset are located. Then from these neighbors the prediction is made.
The similarity between neighbor is measured using Euclidean Distance and Manhattan Distance.
The steps can be summarized as follows:
*Calculate the Euclidean distance/Manhattan distance*
*Get Nearest neighbors*
*Make Predictions*

## k-fold Cross Validation: ##

Cross-validation is used to estimate the skill of a machine learning model on unseen data. Following steps were followed to perform the cross-validation:
* Shuffle the dataset randomly
* Split the dataset into k groups
* For each group:
  * Take the group as a test dataset
  * Take the remaining groups as a training dataset
  * Fit the model on the training dataset and then test it on the test dataset
* Find the evaluation score for the model

Configuration of the k-parameter is an important step in this validation. The value for k is chosen such that each train/test group of data samples is large enough to be statistically representative of the broader dataset. It has been found through experimentation, that setting the value of k = 10 generally results in a model skill estimate with low bias, a modest variance. This value of k can be adjusted to improve the accuracy of the model.
The data from the .csv file is read and stored using pandas library. The dataset is then encoded using OneHotEncoder for converting the values in the dataset from number or string to categorical values.
The distance between the train_row and test_row is found using the vectorized Euclidean distance or the vectorized Manhattan distance. Euclidean distance is the straight-line distance between two points in Euclidean space. Whereas the Manhattan distance is the distance between two points measured along axes at right angles.
The number of k-folds and the number of neighbors are specified in the code and the user is asked which distance is to be calculated for processing. The variation of the value for the number of neighbors and k-folds changes the accuracy of the model. So these values and the seeds are used for the tuning of the model. Based on the k-value selected, the folds are formed and the model is trained and tested on these folds in a loop. In this way the model is evaluated and the accuracy of the model is found. This accuracy is then compared with the accuracy found using the Weka tool.
Weka is a collection of machine learning algorithms for data mining tasks. Weka contains tools for data pre-processing, classification, regression, clustering, association rules, and visualization.

### Datasets: ###
Three datasets are available for this comparison:
* Car dataset
* Hayes-Roth dataset
* Breast-cancer dataset
 
### Conclusion: ###
The accuracy found using the Weka tool and after building the k-NN model for all the mentioned datasets is almost similar when the values for the specified parameters are same for both the evaluations. The distances used for the evaluation of all the datasets are Euclidean distance, Manhattan distance and Minkowski distance.
