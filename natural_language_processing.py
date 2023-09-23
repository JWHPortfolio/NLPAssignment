# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class Sentiment():
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.stopwords = False
        
    def _cleanup(self, data, lines):
        
        corpus = []
        
        if(self.stopwords == False):
            #download the stop words

            self.stopwords = True
        
        for i in range(0, lines):
          # Get words only using regex in re
          
          # Make all lower case
          
          # split into an array of strings necessary to remove stop words
          
          # Collect Stop words
          all_stopwords=
          # Add back in words that may influence the sentiment such as 'not'
          all_stopwords=
          # Stem the words and filter out stopwords
          ps = 
          review = [ps ]
          # Turn back into a string
          review = 
          # Append the string to the total
          corpus.append(review)
        return corpus

    def model_learn(self):
        # Importing the dataset
        dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

        corpus = self._cleanup(dataset, len(dataset))
 
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        
        
        # fit to an array using fit_transform
        
        X = 
        
        Set the label (1 for good, 0 for bad)
        y = 

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # Training the Naive Bayes model on the Training set
        from sklearn.naive_bayes import GaussianNB
        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        
        self.stats =  accuracy_score(y_test, y_pred)
        self.modelLearn = True
        
    def model_infer(self, captureString):
        if(self.modelLearn != True):
            self.model_learn()
            
        # Build 1 entry dictionary similar to Reviews structure with Review:String    
        l = { }
        
        # Convert into a dataframe
        dataOne = 
        
        # Cleanup the dataframe
        oneline = self._cleanup(  )
        
        # Transform the datafame to an array using transform
        XOne = 
        
        # Use classifier to predict the value
        y_pred = self.classifier.predict(XOne)
        
        return y_pred > 0
    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)

if __name__ == '__main__':
        m = Sentiment()
        m.model_learn()
        result = m.model_infer("bad terrible stinks horrible")
        if( result > 0):
            print("Good")
        else:
            print("Bad")
        result = m.model_infer("fantastic wonderful super good")
        if( result > 0):
            print("Good")
        else:
            print("Bad")
            
        print( m.model_stats())
