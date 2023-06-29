# Multi-Label Classification using Codeforces Platform 

CodeForce is a Programming language Competitive platform where Questions are posted, usually its quite challenging to come up with an approach for solving looking at the description. So, tagging for questions is essential for a better user experience. In this machine learning problem, the final goal is to predict the tag based on the Question's textual description, abbreviated as 'problem statement'.

Goal and Objectives:

1)Conduct Explorary Data Analysis on problem tags to analyse number of tags per question,number of words in tags and number of unigrams and multigrams in the problem statement.
2)Conduct Data Pre-processing on problem statement to convert text data to lower case,remove unicode characters , html tags,stop words removal,lemma and stemming.
3)Run and compare various Machine Learning algorithms(Multi-label Logistic Regression,Multi-label Random Forest Classifier and Bilstm with Embedding layer(DL algorithm))by performing hyper-parameter tuning to calculate precision,recall,f1-score,hamming-loss.

#Separate problem difficulty from the problem tags , has prblem difficulty is not considered for tag analysis
#Drop the empty values with empty  problem tags
#Drop the missing values from dataframe
#Calculate number of problem tags in the dataset

Observations:Majority of the most frequent tags are implementation and maths tags.From the above we can conclude that null values are removed from the dataframe.No Duplicate data points that is denoted by the value in 1.
<ul>1.Maximum number of tags per question: 11 </ul>
<ul>2.Minimum number of tags per question: 1 </ul>
<ul>3.Avg. number of tags per question: 2.561 </ul>
<ul>4.Most of the questions are having 2 or 3 tags </ul>


**Model Performance Conclusion**

From the above table we can see that Logistic Regression has the best f1-score compared to random forest regressor and lstm . This may be due to small dataset size and also overfitting. Techniques like oversampling or undersampling can be applied to make the dataset with various tags more balanced. As the dataset is small , machine learning model is performing better than deep learning model. 

However, Bert can be used to improve the f1-score further.


