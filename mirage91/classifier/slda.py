"""
Title: sLDA Classifier
Authors: Mirage 91
"""
# imports python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# main
class sLDA():
    '''
    Classifier class for shrinkage Linear Discriminant Analysis (sLDA).
    '''
    def __init__(self, x, y, train_size=0.8, shrinkage='auto', method='lsqr'): 
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
        self.shrinkage=shrinkage
        self.method=method
        self.classifier = self._create_classifier()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=train_size)

    def _create_classifier(self):
        '''
        Definition of the classifier. In this case the build in sklearn classifier "LinearDiscriminantAnalysis()" is used.
        Input: -
        '''
        return LinearDiscriminantAnalysis()

    def train_and_test(self, plot_cm=True):
        '''
        Training and testing of the classifier. 
        Input:
            - plot_cm: Boolean variable to decide wether the confusion matrix should be plotted. [bool]
        Output: -
        '''
        self.classifier.fit(self.x_train, self.y_train)
        y_predicted = self.classifier.predict(self.x_test)
        acc = accuracy_score(y_true=self.y_test, y_pred=y_predicted)
        conf_matrix = confusion_matrix(y_true=self.y_test, y_pred=y_predicted)

        # Print accuracy
        print(f'The sLDA classifier has a test accuracy of {acc * 100:.2f}%.')

        # Plot confusion matrix
        if plot_cm:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')