# Text Category Classificattion

Text documents are essential as they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus, filtering of the articles into different categories is required. Often the categorization of the articles is conduced manually and retrospectively; thus, causing the waste of time and resources due to this arduous task. Hence, my project purposes is to categorize the text documents into the respective categories.
## Project Description
 Steps involved in this project included:
 1. Data loading 
 - Load the read the dataset using pandas
 2. Data cleaning 
 - Drop duplicated data using pandas
 3. Data preprocessing
 - Modified the text dataset using a RegEx, or Regular Expression, is a sequence of characters that forms a search pattern. RegEx can be used to check if a string        contains the specified search pattern. By using Regex, I had removed few of the characters such as special characters and also standardized the text inside the        dataset in lowercase format.
 - For the target/Y which is the 'subject' column, I used One Hot Encoder. One-hot encoding is a process by which categorical data (such as nominal data) are converted    into numerical features of a dataset. So, the subjects were converted into numerical features so that the model can learn to train it.
 4. Train-test split
 - Train test split is a model validation process that allows me to simulate how the chosen model would perform with my new data.
 5. Model Development(LSTM) & compilation
- Epoch accuracy graph from tensorboard
 <br> ![img](/resources/Epoch_acc.PNG)
 - Epoch loss graph from tensorboard
  <br> ![img](/resources/Capture.PNG)
 6. Evaluation & Prediction
 - Using different evaluation metrics to understand a machine learning model's performance, as well as its strengths and weaknesses.
 <br><br> ![img](/resources/evaluate.PNG)
 

 
  
 ### Acknowledgement 
 1. Source of dataset : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
 2. Source of image : https://www.marscrowd.com/blog/text/text-classification-using-deep-learning/
