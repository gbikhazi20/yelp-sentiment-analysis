# Yelp Sentiment Analysis

### Introduction:
#### Problem Statement:
Computers and even humans have a hard time understanding typed/written messages, since there are no extralinguistic indicators to express emotions or tone (how many times has something like sarcasm been misinterpreted?). Additionally, not all users writing reviews are diligent. Someone may pick ratings for a product without being careful, leading to an inaccurate or inadequate review.

This report aims to better understand the connection between text comments for product reviews and the corresponding ratings given by users. While it may be the case that ratings match up well with the comments, there are bound to be exceptions to this.  Observing any discrepancies between numerical ratings and textual comments could indicate the quality of user reviews.

#### Significance:
All accurate reviews, positive or negative, are useful. However, when there are inconsistencies within a single review, the value of the review drops significantly. Locating discrepancies allows for a more holistic view of a product for both consumers and suppliers. E-commerce companies and their users could potentially elect to filter out these reviews to obtain more accurate representations of the product(s) they are dealing with.

#### Approaches:
Since we are attempting to classify text-based reviews as good or bad, we need to approach the problem with models suited to solve classification problems with sentiment analysis. Therefore, we plan to use some common types of classification algorithms including linear models, generative probabilistic algorithms, tree-based methods, and neural networks. Linear models assume a linear relationship between the input features and target variables.They are simple, fast, and can work well when the decision boundary is linear or when the input features are highly correlated. Linear models can also be easily interpreted and used to gain insight on which features are most important, but they may not be able to map out more complex relationships between the input features and sentiment. Another valid approach is to address the problem with generative probabilistic classification algorithms. These algorithms model the distribution of each class and use Bayes’ theorem to compute the posterior probability of each class based on the input features. Generative probabilistic classification algorithms can provide insight into the underlying data distribution, and they are effective in cases where there is limited training data. However, they may overfit training data and some of these algorithms, Naive Bayes, for example, assume independence between features which may or may not be realistic depending on the dataset. Tree-based methods construct decision trees to recursively divide the input feature space into regions such that each region consists of data points that are similar in terms of the target variable. These methods may not be the most effective for our sentiment analysis problem, but they generally offer a nice set of parameters for hypertuning which can allow for increased accuracy through cross-validation. Neural networks are a class of machine learning algorithms that mimic the structure and function of the human brain. They consist of layers of interconnected nodes that learn to transform the input features into a set of hidden representations that can be used to predict the target variable. Neural networks are generally more computationally expensive, but they generally provide the highest accuracy because of their ability to map complex relationships. We plan to use the Discovery cluster to address the slower runtimes that accompany neural networks. In terms of sentiment analysis, all of these approaches can be used to predict the sentiment of a text based on the frequency of certain words and phrases. In this case, we can predict whether a review is positive or negative based on the number of occurrences of positive or negative words.

### Setup:
We obtained our data from the Yelp Dataset, a collection of datasets provided by Yelp for the purpose of learning NLP.  The particular dataset we chose to use was the review dataset, which contained the texts of customer reviews for various businesses.

The original dataset contained the following 9 columns:
review_id - String - 22 character unique review ID
user_id - String - 22 character unique user ID
business_id - String - 22 character unique business ID
stars - int - star rating
date - String - date formatted as YYYY-MM-DD
text - String - the review itself
useful - int - number of useful votes received
funny - int - number of funny votes received
cool - int - number of cool votes received

For the purposes of this analysis, the only feature column used was text. This feature was used to predict target variable stars.

The following chart displays review frequencies for each category on a randomly sampled subset of the dataset of 50,000 samples (`data/yelp_academic_dataset_review_50k.txt`). As you can see, there are significantly more 5 star reviews than any other category. 

<div align=center><img src='./figures/review_distribution.png' width="500"></div>

The neural models were trained on subsets of the entire dataset, so they might experience some bias as a result of the skewed distribution. The statistical models, not needing as much data, were trained on subsets of `data/yelp_academic_dataset_review_50k.txt`. Some of these subsets were built out using the `build_subset()` function which is able to produce versions of the dataset with balanced review frequencies for each category.

`build_subset()` allowed us to build different versions of the dataset with different distributions. It also allowed us to create datasets with any combination of review categories, e.g. only 1 and 5-star reviews, only 1, 2, 4, and 5- star reviews, etc. We tested our models accordingly, either predicting between two groups (1 and 5 stars, 1 or 2 and 4 or 5 stars, etc.) or multiple.

Various statistical models were evaluated, namely Naive Bayes, Logistic Regression, KNN, LDA, QDA, and SVM. For these models we used implementations provided by the [scikit-learn](https://scikit-learn.org/) library and in some cases also wrote up our own implementations (this was done for the Naive Bayes model and the Logistic Regression model). These models are all light-weight and were run locally on our personal machines.

Two neural models were assessed, a [pre-trained BERT](https://huggingface.co/bert-base-cased) provided by Hugging Face and a convolutional neural network trained from scratch. The BERT model is based on the Transformer architecture. The CNN was architected to have 3 filters The pre-trained model was fine-tuned on a small subset of the data (3,000 samples) and the CNN was trained on 487,500 samples. Both networks were trained on Northeastern University's high-performance computing resource, the Discovery cluster. 

