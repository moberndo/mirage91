# Coding standard
To ensure that we work with a nice code, I would like to propose a coding standard everybody should follow.
Here are some more general guidelines:
- If you define functions inside a class that are only used inside the class, start the function name with a _.
- Don't access varibles of a class directly. Always access variables via a function.
- The structure of every file should be:
1) title and author of the file
2) all imports, first built-in python functions, then custom functions (if necessary)
3) settings: whatever can be changed/set should be done in the first few lines. There should not be one input on line 14 and another on line 182 and another on line 413.
4) then all functions (if necessary)
5) then the main part of the file
(for reference, look at the files that are uploaded so far)
- Don't forget to write comments

Some guidlines relevant for this project:
The EEG processing should work indepndently from the classification. If you want to make changes, feel free to do so, but always make sure that the output of the processing part is:
- a feature vector
- a target vector
This way we can work on the processing and classification simultaniously.

# ToDo
## First part: Signal processing
The processing does not need any immanent changes.

## Second part: Feature extraction
In this basic version I only looked at the bandpower of each channel for different frequency bands. Another apporoach could be the central spatial pattern (CSP)

## Third part: Classification
For now only sLDA is implemented. However, it still needs cross-validation for more stable test-accuracies, since we dont have too much training data.
Add/Test classifiers in whatever shape and form you prefer them, just make sure that they have a "train_and_test()" function, this way we dont need to make many changes in the main file.
Also put every classifier into a seperate .py file, which should then be stored in the 'classifier' folder.
