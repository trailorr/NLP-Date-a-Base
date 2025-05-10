# NLP-Date-a-Base
NLP Final Project by The Naive Babes

The directory is composed of the CoLab Date-A-Base.ipynb file, a folder containing the original dataset, and a copy of our final slides. 

The dataset is split in the file as the original was too large to be uploaded to GitHub as one submission. Here is a link to the original Kaggle post containing the full dataset:
https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles

The code may be executed directly from CoLab as it pulls the dataset directly from the GitHub repo (no need to download dataset). 

In addition to standard Python libraries, we used the [Mistral](https://docs.mistral.ai/getting-started/clients/) library as well as the [HuggingFace](https://huggingface.co/docs/hub/en/transformers) libraries used in Lab 7 for DistiliBERT.

# Breakdown of the CoLab
First, pull and combine the datasets from this repository. 

## Prep data
Partition the data into 80% training data, 20% testing data. Then extract only the columns that will be used, the age columns and the essay columns. Then to figure out the size of each age range, entries were grouped according to the user's age and then the groups were counted. Lastly, essay responses from multiple columns were combined into one string for processing. 

## Baselines
Both training and testing datasets were balanced with 2000 entries per age group. Then baselines were calculated: random, majority, and stratified. 

## Bag of Words with POS
First, tokenized and tagged both training and testing data. Then frequencies of each essay were calculated. Next, created bag of words vectors with part of speech lists and frequencies. Lastly, trained the bag of words model on the training data and tested it on the testing data. The results of the model were then recorded.

## Word2Vec with SVM
First, all essay responses were preprocessed to one string and tokenized. Then the Word2Vec model was created using the sentence tokens. Essays were then converted to feature vectors for both training and testing datasets. Then the svm was fitted using the training vectors. Lastly, the svm was used to predict the testing data. The results of the model were then recorded.

## TF-IDF
First, TF-IDF model was initialized with both unigrams and bigrams. The model was then trained with the RandomForestClassifier. Lastly, the model was tested with the testing data. The results of the model were then recorded.

## Mistral (zero-shot)
The model "mistral-large-latest" was used. Using 200 testing examples, the LLM was given a prompt without examples asking to classify the essay responses into four age groups. To not run out of usage for the free plan, a sleep period of five seconds was included. The results of the model were then recorded.

## Mistral (few-shot)
The model "mistral-large-latest" was used again. Using 200 testing examples, the LLM was given a prompt with two examples of a 24 year old response, and a 39 year old response. The model was asked to classify the essay responses into four age groups. To not run out of usage for the free plan, a sleep period of five seconds was included. The results of the model were then recorded.

## BERT
First, encodings were created for both training and testing sets that were tokenized. Datasets objects were created with training and testing data. Then the DistilBertForSequenceClassification model was fine-tuned with the training dataset object. Lastly, the model was applied to the testing dataset. The results of the model were then recorded.

## Comparing the models
Using matplotlib, the results of all models, as well as the baselines calculated in the beginning of the CoLab were compared and displayed in a bar chart. 
