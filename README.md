# Data-visualization

## Project Overview

This project consists of four parts aimed at improving previous models and applying data science techniques to various datasets. The project highlights model improvement, dimensionality reduction, and image classification tasks.

## Part 1: Improving Last Semester's Model

Recap of Previous Model
In the previous semester, I built a multi-class classification model that predicts the year a survey was taken based on multiple features. I combined five surveys conducted between 2015 and 2019 into one dataset, adding a 'year' column and removing any data that was not common to all five surveys. The features used in the model were:

Overall rank: the rank of the year among all countries
Country or region: the country that was ranked
Score: the overall score
GDP per capita: Gross domestic product, a measure of economic well-being
Social support: the score for social support
Healthy life expectancy: the score for life expectancy
Freedom to make life choices: the score for freedom
Generosity: the score on the survey for generosity
Perceptions of corruption: the score for perceptions of corruption
Year: the year the survey was conducted
The model achieved an accuracy rate of around 45%. This accuracy was expected, as there is not much variation in the survey results from year to year. However, it still performed better than the "dummy model" with an accuracy of 19.7%.

Improvements Made
To further improve the model's accuracy, I made the following improvements based on what I learned during this semester:

Used an Ensemble Model XGBoost: The new model achieved an accuracy of 49.12%, outperforming the previous model.

Data Normalization and Dimensionality Reduction: I normalized the data using standard scaler and applied Principal Component Analysis (PCA) to reduce the dimensions.

Reduced to 3 dimensions and plotted the results.
Further reduced to 2 dimensions and plotted the results.
The PCA analysis showed that by reducing the dimensions, the model achieved a close accuracy of 42% while using only 2 features. Additionally, the new ensemble model was able to achieve a nearly perfect accuracy of 52% with just two features, matching its performance with 4 features.

Conclusion
By using a better model, XGBoost, I improved the accuracy of the model from 45% to 52% while reducing the number of dimensions from 4 to 2. This improvement demonstrates the effectiveness of the ensemble model and the benefits of dimensionality reduction techniques like PCA. It's a double win for achieving higher accuracy with fewer features and dimensions.

## Part 2: Fashion Mnist
Project Overview
In this part of the project, I focused on the Fashion MNIST dataset. Here is a summary of the steps I followed:

Importing Required Libraries and Loading the Data:

Imported pandas, numpy, and matplotlib libraries.
Loaded the data into a pandas DataFrame.
Reviewed the first five rows of the data to get an initial understanding.
Data Preprocessing and Initial Model:

Split the data into features and labels.
Reviewed the first 15 images from the dataset.
Created a KNN model without performing data normalization and PCA for comparison purposes.
Data Normalization and PCA:

Normalized the data to ensure uniform scales across features.
Conducted Principal Component Analysis (PCA) to reduce the dimensionality of the data.
Plotted the explained variance as a function of the number of principal components (PCs).
Chose to reduce the 784 features to 50 principal components.
Model Training and Evaluation:

Defined and trained KNN and XGBoost models on the reduced-dimension training data.
Imported and cleaned the test data to evaluate the model accuracy.
Evaluated the accuracy of the KNN model before PCA (approximately 85.89%).
Transformed the testing data using the same PCA components.
Evaluated the accuracy of models trained on the reduced training data (KNN: 85.46%, XGBoost: 87.04%).
Conducted cross-validation and obtained the cross-validation scores for both models (KNN: 85.47%, XGBoost: 87.04%).
Model Comparison:

Compared the performance of the models to a dummy classifier (accuracy: 10%).
Concluded that the KNN model achieved an accuracy of 85.89% before normalization and PCA, 85.46% after PCA with 50 dimensions, and the XGBoost model achieved an accuracy of 87.04% after PCA with 50 dimensions.
Highlighted that using 50 dimensions yielded nearly as good results as using all 784 dimensions, demonstrating the effectiveness of dimensionality reduction techniques.
Outperformed the dummy classifier, indicating that the model has learned meaningful patterns from the data.
By employing data normalization, PCA for dimensionality reduction, and utilizing ensemble models like XGBoost, I achieved high accuracy in classifying the Fashion MNIST dataset while significantly reducing the number of dimensions.

## Part 3: Dogs vs. Cats
Project Overview
In this part of the project, I focused on classifying images of dogs and cats. Here is a summary of the steps I followed:

Importing Required Packages and Loading the Data:

Imported necessary packages such as cv2, pandas, numpy, and matplotlib.
Loaded single cat and dog images.
Data Preprocessing and Initial Model:

Resized the images to 10x10 pixels initially, but found the images to be unrecognizable.
Resized all images to a more recognizable size of 30x30 pixels.
Created separate data frames for cats and dogs and merged them.
Conducted data cleaning and preparation.
Reviewed the first 15 images from the dataset.
Split the data into features and labels.
Created initial models, such as KNN and DecisionTreeClassifier, before performing data normalization and PCA.
Evaluated the accuracy of the models before normalization and PCA (KNN: 0.6062, DecisionTreeClassifier: 0.5572, XGBoost: 0.6496).
Data Normalization and PCA:

Normalized the data to ensure uniform scales across features.
Applied Principal Component Analysis (PCA) to reduce the dimensionality of the data.
Plotted the top 10 principal components as a function of variance explained.
Model Training and Evaluation:

Chose the number of components, d, based on the explained variance ratio obtained from PCA.
Trained and evaluated the models using the reduced-dimension training data.
Evaluated the accuracy of the models after PCA with 188 dimensions (KNN: 0.4992, DecisionTreeClassifier: 0.4944, XGBoost: 0.5172).
Model Comparison:

Compared the performance of the models to a dummy classifier (accuracy: 0.1).
Concluded that the KNN and DecisionTreeClassifier models performed worse after dimensionality reduction.
Noted that the XGBoost model achieved an accuracy of 65.46% without dimensionality reduction, indicating its effectiveness in classifying dog vs. cat images.
Despite the challenges in reducing dimensions and achieving high accuracy, the XGBoost model proved to be a strong classifier without dimensionality reduction. It achieved an accuracy rate of 65.46% in distinguishing between dog and cat images.

## Part 4: Hand

### Introduction

In this part of the project, the focus was on developing a classifier for hand gestures. The key idea was to treat the data as a video rather than individual images to capture the dynamics of hand movements. Here is an overview of the steps involved:

1. Understanding Image vs Video Classification:
   - Explained the difference between image and video classifiers using an example of a moving ball.
   - Emphasized the importance of considering a series of frames (images) to predict the direction of motion accurately.

2. Data Preparation:
   - Imported necessary files and split them into separate lists: "Alone," "Sync," and "Spontaneous."
   - Merged the right-hand files with the "Alone" files.
   - Removed the first 7 seconds and two features from the dataset.
   - Applied similar steps for the "Sync" and "Spontaneous" files.
   - Reduced each file by taking every 8th frame to account for slow human movements.
   - Created rows with 10 reduced frames, representing approximately 1 second of the hand video.
   - Assigned labels: 1 for "Alone," 2 for "Sync," and 3 for "Spontaneous."
   - Merged all datasets into one.

3. Dataset Analysis:
   - Analyzed the standard deviation and mean values of each feature to gain insights into the data.
   - Observed that "Alone" gestures were distinct, while "Sync" and "Spontaneous" gestures were more similar.

4. Model Development and Evaluation:
   - Transformed the dataset into videos rather than individual images.
   - Checked the performance of a dummy classifier as a benchmark.
   - Built KNN and XGBoost models.
   - Modified the testing data similarly to the training data.
   - Evaluated the models on the testing data.
   - Achieved an accuracy of 85.20% for the KNN model and 89.68% for the XGBoost model.
   - Performed k-fold validation and obtained a mean accuracy of 90.25% for KNN and 94.77% for XGBoost.

5. Model Analysis:
   - Verified that the models performed well in predicting the "Alone" gestures with high accuracy.
   - Noted that distinguishing between "Sync" and "Spontaneous" gestures was more challenging.
   - Visualized the confusion matrix using Seaborn.

Overall, the developed models demonstrated strong performance in classifying hand gestures, achieving an accuracy of 89.68% with the XGBoost model. The results validated the effectiveness of treating the hand gesture data as videos rather than individual images.

## Technologies Used

- Python
- XGBoost
- KNN
- PCA

