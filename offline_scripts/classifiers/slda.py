"""
Title: sLDA Classifier
Authors: Mirage 91
"""
# imports python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# main
class sLDA:
    '''
    Classifier class for shrinkage Linear Discriminant Analysis (sLDA).
    '''
    #def __init__(self, x, y, train_size=0.8, shrinkage='auto', method='lsqr'): 
    def __init__(self, x_train, x_test, y_train, y_test, shrinkage='auto', method='lsqr'):
        '''
        Initialize the class with a feature and target vector.
        Input:
            - x: Feature vector [Numpy array]
            - y: Target vector [Numpy array]
            - train_size: Training size for classifier to be trained on. Has to be a
            float 0-1. The test size is then calculated as 1-training_size [float]
            - shrinkage: ...
            - method: ...
        '''
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.shrinkage=shrinkage
        self.method=method
        self.classifier = self._create_classifier()

    def _create_classifier(self):
        '''
        Definition of the classifier. In this case the build in sklearn classifier "LinearDiscriminantAnalysis()" is used.
        Input: -
        '''
        return LinearDiscriminantAnalysis()


    def train_and_test(self, plot_cm=True, modality=None):
        '''
        Training and testing of the classifier. 
        Input:
            - plot_cm: Boolean variable to decide whether the confusion matrix should be plotted. [bool]
            - modality: Optional string to add to the confusion matrix title. [str]
        Output: -
        '''
        
        # Train the classifier
        self.classifier.fit(self.x_train, self.y_train)
        y_predicted = self.classifier.predict(self.x_test)
        acc = accuracy_score(y_true=self.y_test, y_pred=y_predicted)
        
        # Define all possible labels
        conf_matrix = confusion_matrix(y_true=self.y_test, y_pred=y_predicted)
        
        # Print accuracy
        print(f'The classifier has a test accuracy of {acc * 100:.2f}%.')
    
        # Plot confusion matrix
        if plot_cm:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels= ['left_foot', 'left_hand', 'mental_singing', 'right_foot'],
                        yticklabels= ['left_foot', 'left_hand', 'mental_singing', 'right_foot'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix {modality}')
            plt.savefig('confusion_matrix.png')
            plt.show()
            
   


    
    