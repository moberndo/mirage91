# '''
# Pipeline Testing Script with built in mne methods and functions
# '''
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.preprocessing import FunctionTransformer, StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from statsmodels.tsa.ar_model import AutoReg
# import pywt
# from scipy.stats import skew, kurtosis
# import matplotlib.pyplot as plt
# from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
# from sklearn.base import BaseEstimator, TransformerMixin

# # Reshape for Autoencoder
# class ReshapeForScaler(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         # Flatten the time steps and features into a single feature dimension
#         num_samples = X.shape[0]
#         num_features = X.shape[1] * X.shape[2]
#         return X.reshape(num_samples, num_features)

# class ReshapeBack(BaseEstimator, TransformerMixin):
#     def __init__(self, original_shape):
#         self.original_shape = original_shape
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         num_samples = X.shape[0]
#         return X.reshape(num_samples, self.original_shape)
    
# # Define PyTorch Autoencoder model
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_dim)  # Ensure this matches the flattened dimension
#         )

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # Define PyTorch Autoencoder wrapper for scikit-learn
# class PyTorchAutoencoder(BaseEstimator, ClassifierMixin):
#     def __init__(self, input_dim, epochs=50, batch_size=32):
#         self.input_dim = input_dim
#         self.epochs = epochs
#         self.batch_size = batch_size
        

#     def fit(self, X, y=None):
#         self.model = Autoencoder(input_dim=self.input_dim)
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.scaler = torch.amp.GradScaler('cuda') # Mixed precision training
        
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         for epoch in range(self.epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
#             with torch.amp.autocast('cuda'):
#                 outputs = self.model(X_tensor)
#                 loss = self.criterion(outputs, X_tensor)
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#         return self

#     def transform(self, X):
#         self.model.eval()
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         with torch.no_grad():
#             encoded = self.model.encoder(X_tensor).cpu().numpy()
#         return encoded

#     def predict(self, X):
#         # Placeholder method, not used in current pipeline setup
#         return np.zeros(X.shape[0])


# # Define the class with different pipelines
# class AdvancedPipelineEvaluator:
#     def __init__(self, X_train, X_test, y_train, y_test):
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.results = {}

#     def wavelet_features(self, X):
#         # List to store flattened wavelet coefficients
#         features_list = []
#         for x in X:
#             # Perform wavelet decomposition
#             coeffs = pywt.wavedec(x, 'db4', level=5)
#             # Flatten each coefficient array and concatenate them
#             flattened_coeffs = [coeff.flatten() for coeff in coeffs]
#             # Concatenate all flattened coefficients
#             concatenated_features = np.concatenate(flattened_coeffs)
#             features_list.append(concatenated_features)
#         return np.array(features_list)

#     def fourier_features(self, X):
#         fourier_result =  np.abs(np.fft.fft(X))         
#         if len(fourier_result.shape) == 3:
#             n_samples, n_features, n_something = fourier_result.shape
#             fourier_result = fourier_result.reshape((n_samples, n_features * n_something))
#         return fourier_result

#     def autoregressive_features(self, X):
#         # Ensure X has the expected shape (n_samples, n_features, n_timesteps)
#         if len(X.shape) != 3:
#             raise ValueError(f'Expected 3D array for X, but got shape {X.shape}')
#         n_samples, n_features, n_timesteps = X.shape
#         # Initialize an empty list to store autoregressive features
#         autoregressive_features_list = []
#         for feature_idx in range(n_features):
#             # Extract each feature across all samples
#             feature_data = X[:, feature_idx, :]    
#             # Apply AutoReg to each feature separately
#             for sample_idx in range(n_samples):
#                 sample_data = feature_data[sample_idx, :]
#                 model = AutoReg(sample_data, lags=5)
#                 model_fit = model.fit()
#                 fitted_values = model_fit.fittedvalues
#                 # Append fitted values
#                 autoregressive_features_list.append(fitted_values)
#         # Convert list of fitted values to array and reshape
#         autoregressive_features_array = np.array(autoregressive_features_list).reshape(n_samples, n_features, -1)
#         # Flatten the 3D array to 2D
#         autoregressive_features_flat = autoregressive_features_array.reshape(n_samples, -1)    
#         return autoregressive_features_flat

#     def skewness_kurtosis_features(self, X):
#         # Calculate skewness and kurtosis per sample
#         skewness = skew(X.reshape(X.shape[0], -1), axis=1)
#         kurt = kurtosis(X.reshape(X.shape[0], -1), axis=1)
#         # Combine skewness and kurtosis into a single feature matrix
#         features = np.concatenate((skewness.reshape(-1, 1), kurt.reshape(-1, 1)), axis=1)
#         return features

#     def build_pipeline_wavelet(self):
#         return Pipeline([
#             ('wavelet', FunctionTransformer(self.wavelet_features, validate=False)),
#             ('scaler', StandardScaler()),
#             ('classifier', GradientBoostingClassifier())
#         ])

#     def build_pipeline_fourier(self):
#         return Pipeline([
#             ('fourier', FunctionTransformer(self.fourier_features, validate=False)),
#             ('scaler', StandardScaler()),
#             ('classifier', RandomForestClassifier())
#         ])
    
#     def build_pipeline_autoregressive(self):
#         return Pipeline([
#             ('autoregressive', FunctionTransformer(self.autoregressive_features, validate=False)),
#             ('scaler', StandardScaler()),
#             ('classifier', SVC())
#         ])

#     def build_pipeline_skewness_kurtosis(self):
#         return Pipeline([
#             ('skew_kurt', FunctionTransformer(self.skewness_kurtosis_features, validate=False)),
#             ('scaler', StandardScaler()),
#             ('classifier', LinearDiscriminantAnalysis())
#         ])

#     def build_pipeline_autoencoder(self):
#         num_features = self.X_train.shape[1] * self.X_train.shape[2]  # Flattened dimension
#         return Pipeline([
#             ('reshape_for_scaler', ReshapeForScaler()),
#             ('scaler', StandardScaler()),
#             ('reshape_back', ReshapeBack(original_shape=(num_features))),
#             ('autoencoder', PyTorchAutoencoder(input_dim=num_features, epochs=50, batch_size=32)),
#             ('classifier', SVC())
#         ])

#     def build_and_evaluate_model(self, pipeline_fn):
#         pipeline = pipeline_fn()

#         param_grid = {
#             'classifier__C': [0.1, 1, 10] if isinstance(pipeline.named_steps['classifier'], SVC) else [None],
#             'classifier__n_estimators': [100, 200] if isinstance(pipeline.named_steps['classifier'], (GradientBoostingClassifier, RandomForestClassifier)) else [None],
#             'classifier__learning_rate': [0.01, 0.1, 0.2] if isinstance(pipeline.named_steps['classifier'], GradientBoostingClassifier) else [None],
#             'classifier__max_depth': [3, 4, 5] if isinstance(pipeline.named_steps['classifier'], GradientBoostingClassifier) else [None],
#         }
        
#         #print(f"Shape of X_train: {self.X_train.shape}")
#         #print(f"Shape of y_train: {self.y_train.shape}")
        
#         # Remove parameters that are not relevant for the current classifier
#         param_grid = {k: v for k, v in param_grid.items() if v != [None]}
#         print('START GRID SEARCH')
#         grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, pre_dispatch=2, error_score='raise')
#         #print("Fitting grid search...")
#         grid_search.fit(self.X_train, self.y_train)
#         print('END GRID SEARCH')
#         best_model = grid_search.best_estimator_
#         best_params = grid_search.best_params_
#         y_pred = best_model.predict(self.X_test)
#         accuracy = accuracy_score(self.y_test, y_pred)
#         report = classification_report(self.y_test, y_pred)

#         print(f'Best Parameters: {best_params}')
#         print(f'Accuracy: {accuracy}')
#         print(f'Classification Report:\n{report}')

#         # Plot confusion matrix with TEST data
#         # Generate the confusion matrix display
#         disp = ConfusionMatrixDisplay.from_estimator(
#             best_model, 
#             self.X_test, 
#             self.y_test,
#             cmap=plt.cm.Reds  # Optional: choose a colormap
#         )
#         # Extract the names of the components from the best_model pipeline
#         feature_method_name = best_model.named_steps.__class__.__name__
#         #print(feature_method_name)
#         scaler_name = best_model.named_steps['scaler'].__class__.__name__
#         #print(scaler_name)
#         classifier_name = best_model.named_steps['classifier'].__class__.__name__
#         #print(classifier_name)
        
#         # Add a title
#         disp.ax_.set_title(f'Confusion Matrix with \n {feature_method_name} \n {scaler_name} \n {classifier_name}')
        
#         # Customize the labels in correct order  - Run EEG class (self.epochs.event_id)
#         labels = ['left_foot', 'left_hand', 'mental_singing', 'right_foot']  # Modify as per your class labels
#         #disp.ax_.set_xticklabels(labels, rotation=0)
#         #disp.ax_.set_yticklabels(labels)
        
#         # Add a description or any additional text
#         plt.figtext(0.5, 0.01, f'{best_params}', ha='center', va='center', fontsize=12)
        
#         # Show the plot
#         #plt.show()

#         return best_model, best_params

#     def evaluate_all_pipelines(self):
#         pipelines = [
#             #self.build_pipeline_wavelet, # needs much time
#             self.build_pipeline_fourier,
#             self.build_pipeline_autoregressive,
#             self.build_pipeline_skewness_kurtosis,
#             self.build_pipeline_autoencoder
#         ]

#         for pipeline_fn in pipelines:
#             print(f"Testing pipeline: {pipeline_fn.__name__}")
#             best_model, best_params = self.build_and_evaluate_model(pipeline_fn)
#             self.results[pipeline_fn.__name__] = (best_model, best_params)
'''
Pipeline Testing Script with built in mne methods and functions
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib  # For saving and loading models
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.tsa.ar_model import AutoReg
import pywt
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

# Reshape for Autoencoder
class ReshapeForScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_samples = X.shape[0]
        num_features = X.shape[1] * X.shape[2]
        return X.reshape(num_samples, num_features)

class ReshapeBack(BaseEstimator, TransformerMixin):
    def __init__(self, original_shape):
        self.original_shape = original_shape
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_samples = X.shape[0]
        return X.reshape(num_samples, self.original_shape)
    
# Define PyTorch Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Ensure this matches the flattened dimension
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define PyTorch Autoencoder wrapper for scikit-learn
class PyTorchAutoencoder(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        

    def fit(self, X, y=None):
        self.model = Autoencoder(input_dim=self.input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = torch.amp.GradScaler('cuda')  # Mixed precision training
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, X_tensor)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return self

    def transform(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.model.encoder(X_tensor).cpu().numpy()
        return encoded

    def predict(self, X):
        # Placeholder method, not used in current pipeline setup (only for encoding used)
        return np.zeros(X.shape[0])


'''
PARAMETERS in PIPELINE:
    - different features obtain information of different parts of the signal (time, frequency, non-linear)
    - PCA: reduces components and therefore dimensions and complexity of data
    - Scaler: 
        + StandardScaler: standardizes features by removing the mean and scaling to unit variance. 
        + MinMaxScaler: scales features to a fixed range, typically [0, 1].
        + RobustScaler: scales features based on the median and interquartile range (IQR).
        + MaxAbsScaler: scales features to a fixed range, typically [-1, 1].
        + PowerTransformer: applies a power transformation to make the data more Gaussian-like.
        + QuantileTransformer: transforms features to follow a uniform or normal distribution.





'''
# Define the class with different pipelines
class AdvancedPipelineEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}

    # def wavelet_features(self, X): # provides a time-frequency representation of the signal
    #     features_list = []
    #     for x in X:
    #         coeffs = pywt.wavedec(x, 'db4', level=5)
    #         flattened_coeffs = [coeff.flatten() for coeff in coeffs]
    #         concatenated_features = np.concatenate(flattened_coeffs)
    #         features_list.append(concatenated_features)
    #     return np.array(features_list)
    # def wavelet_features(self, X): # provides a time-frequency representation of the signal
    #     def extract_features(x):
    #         coeffs = pywt.wavedec(x, 'db4', level=5)
    #         flattened_coeffs = [coeff.flatten() for coeff in coeffs]
    #         concatenated_features = np.concatenate(flattened_coeffs)
    #         return concatenated_features
    #     # Parallelize the computation
    #     features_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(extract_features)(x) for x in X)
    #     return np.array(features_list)
    
    def bandpower_features(self, X, freq_bands=[(3, 5), (10, 15)]): # captures relationship in frequency domain
        features_list = []
        for x in X:
            bandpower_features = []
            for band in freq_bands:
                low_freq, high_freq = band
                # Compute the power spectral density
                freqs, psd = signal.welch(x, fs=250, nperseg=256, axis=-1)  # 250 Hz sampling frequency and 256-point FFT
                # Integrate the PSD over the frequency band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                bandpower = np.trapz(psd[:, band_mask], freqs[band_mask], axis=-1)
                bandpower_features.append(bandpower)
            features_list.append(np.concatenate(bandpower_features))
        return np.array(features_list)
    
    def fourier_features(self, X): # captures relationship in frequency domain
        fourier_result =  np.abs(np.fft.fft(X))         
        if len(fourier_result.shape) == 3:
            n_samples, n_features, n_something = fourier_result.shape
            fourier_result = fourier_result.reshape((n_samples, n_features * n_something))
        return fourier_result

    def autoregressive_features(self, X): # captures relationship between time points
        if len(X.shape) != 3:
            raise ValueError(f'Expected 3D array for X, but got shape {X.shape}')
        n_samples, n_features, n_timesteps = X.shape
        autoregressive_features_list = []
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx, :]    
            for sample_idx in range(n_samples):
                sample_data = feature_data[sample_idx, :]
                model = AutoReg(sample_data, lags=5)
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues
                autoregressive_features_list.append(fitted_values)
        autoregressive_features_array = np.array(autoregressive_features_list).reshape(n_samples, n_features, -1)
        autoregressive_features_flat = autoregressive_features_array.reshape(n_samples, -1)    
        return autoregressive_features_flat

    def skewness_kurtosis_features(self, X): # measure the asymmetry and peakedness of the distribution
        skewness = skew(X.reshape(X.shape[0], -1), axis=1)
        kurt = kurtosis(X.reshape(X.shape[0], -1), axis=1)
        features = np.concatenate((skewness.reshape(-1, 1), kurt.reshape(-1, 1)), axis=1)
        return features

    def hjorth_parameters(self, X): # measures activity, mobility, and complexity of the signal
        def compute_hjorth_params(data):
            diff1 = np.diff(data, n=1)
            diff2 = np.diff(data, n=2)
            var_zero = np.var(data)
            var_diff1 = np.var(diff1)
            var_diff2 = np.var(diff2)
            activity = var_zero
            mobility = np.sqrt(var_diff1 / var_zero)
            complexity = np.sqrt(var_diff2 / var_diff1) / mobility
            return np.array([activity, mobility, complexity])
        
        features_list = []
        for x in X:
            hjorth_params = np.apply_along_axis(compute_hjorth_params, axis=-1, arr=x)
            features_list.append(hjorth_params.flatten())
        return np.array(features_list)

    def fractal_dimension(self, X): # measures the complexity and self-similarity of the signal
        def compute_fractal_dimension(data):
            N = len(data)
            L = []
            for k in range(1, N // 2):
                Lk = np.sum(np.abs(data[:-k] - data[k:])) * (N - 1) / (2 * k * N)
                L.append(Lk)
            return np.log(np.mean(L)) / np.log(2)
        
        features_list = []
        for x in X:
            fractal_dims = np.apply_along_axis(compute_fractal_dimension, axis=-1, arr=x)
            features_list.append(fractal_dims)
        return np.array(features_list)

    # def build_pipeline_wavelet(self):
    #     return Pipeline([
    #         ('wavelet', FunctionTransformer(self.wavelet_features, validate=False)),
    #         ('scaler', StandardScaler()),
    #         ('classifier', GradientBoostingClassifier())
    #     ])
    
    def build_pipeline_bandpower(self): 
       return Pipeline([
           ('bandpower', FunctionTransformer(self.bandpower_features, validate=False)),
           ('pca', PCA(n_components=50)),
           ('scaler', StandardScaler()),
           ('classifier', GradientBoostingClassifier())
       ])
   
    def build_pipeline_fourier(self): 
        return Pipeline([
            ('fourier', FunctionTransformer(self.fourier_features, validate=False)),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(class_weight='balanced'))
        ])
    
    def build_pipeline_autoregressive(self): 
        return Pipeline([
            ('autoregressive', FunctionTransformer(self.autoregressive_features, validate=False)),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('classifier', SVC(class_weight='balanced'))
        ])

    def build_pipeline_skewness_kurtosis(self):
        return Pipeline([
            ('skew_kurt', FunctionTransformer(self.skewness_kurtosis_features, validate=False)),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('classifier', LinearDiscriminantAnalysis())
        ])

    def build_pipeline_autoencoder(self):
        num_features = self.X_train.shape[1] * self.X_train.shape[2]
        return Pipeline([
            ('reshape_for_scaler', ReshapeForScaler()),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('reshape_back', ReshapeBack(original_shape=(num_features))),
            ('autoencoder', PyTorchAutoencoder(input_dim=num_features, epochs=50, batch_size=32)),
            ('classifier', SVC(class_weight='balanced'))
        ])

    def build_pipeline_hjorth(self): 
        return Pipeline([
            ('hjorth', FunctionTransformer(self.hjorth_parameters, validate=False)),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier())
        ])

    def build_pipeline_fractal(self):
        return Pipeline([
            ('fractal', FunctionTransformer(self.fractal_dimension, validate=False)),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(class_weight='balanced'))
        ])

    def build_pipeline_combined_features(self):
        base_learners = [
            ('svc', SVC(probability=True, class_weight='balanced')),
            ('gbc', GradientBoostingClassifier())
        ]
        return Pipeline([
            ('features', FeatureUnion([
                #('wavelet', FunctionTransformer(self.wavelet_features, validate=False)),
                ('bandpower', FunctionTransformer(self.bandpower_features, validate=False)),
                ('fourier', FunctionTransformer(self.fourier_features, validate=False)),
                ('hjorth', FunctionTransformer(self.hjorth_parameters, validate=False)),
                ('fractal', FunctionTransformer(self.fractal_dimension, validate=False)),
            ])),
            ('pca', PCA(n_components=50)),
            ('scaler', StandardScaler()),
            ('ensemble', StackingClassifier(
                estimators=base_learners,
                final_estimator=RandomForestClassifier()
            ))
        ])

    def build_and_evaluate_model(self, pipeline_fn):
        pipeline = pipeline_fn()
                
        param_grid = {
            'classifier__C': [0.1, 1, 10] if isinstance(pipeline.named_steps['classifier'], SVC) else [None],
            'classifier__n_estimators': [100, 200] if isinstance(pipeline.named_steps['classifier'], (GradientBoostingClassifier, RandomForestClassifier)) else [None],
            'classifier__learning_rate': [0.01, 0.1, 0.2] if isinstance(pipeline.named_steps['classifier'], GradientBoostingClassifier) else [None],
            'classifier__max_depth': [3, 4, 5] if isinstance(pipeline.named_steps['classifier'], GradientBoostingClassifier) else [None],
        }

        param_grid = {k: v for k, v in param_grid.items() if v != [None]}
        
       
        print('START GRID SEARCH')
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, pre_dispatch=2, error_score='raise')
        grid_search.fit(self.X_train, self.y_train)
        print('END GRID SEARCH')
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)

        print(f'Best Parameters: {best_params}')
        print(f'Accuracy: {accuracy}')
        print(f'Classification Report:\n{report}')

        disp = ConfusionMatrixDisplay.from_estimator(
            best_model, 
            self.X_test, 
            self.y_test,
            cmap=plt.cm.Reds
        )

        feature_method_name = pipeline_fn.__name__
        scaler_name = best_model.named_steps['scaler'].__class__.__name__
        classifier_name = best_model.named_steps['classifier'].__class__.__name__

        disp.ax_.set_title(f'Confusion Matrix with \n {feature_method_name} \n {scaler_name} \n {classifier_name}')

        labels = ['left_foot', 'left_hand', 'mental_singing', 'right_foot']
        
        plt.figtext(0.5, 0.01, f'{best_params}', ha='center', va='center', fontsize=12)

        plt.show()

        return best_model, best_params, accuracy

    def load_last_saved_model(self, file_path='best_model.pkl'):
        try:
            return joblib.load(file_path)
        except FileNotFoundError:
            return None

    def save_model(self, model, file_path='best_model.pkl'):
        joblib.dump(model, file_path)

    def build_evaluate_and_compare(self, pipeline_fn, file_path='best_model.pkl'):
        best_model, best_params, current_accuracy = self.build_and_evaluate_model(pipeline_fn)
        
        last_saved_model_info = self.load_last_saved_model(file_path)
        if last_saved_model_info:
            last_saved_model, last_saved_accuracy = last_saved_model_info
            if current_accuracy > last_saved_accuracy:
                self.save_model((best_model, current_accuracy), file_path)
                print(f'New model saved with accuracy: {current_accuracy}')
            else:
                print(f'Last saved model retained with accuracy: {last_saved_accuracy}')
        else:
            self.save_model((best_model, current_accuracy), file_path)
            print(f'No previous model found. Current model saved with accuracy: {current_accuracy}')

    def evaluate_all_pipelines(self):
        pipelines = [
            #self.build_pipeline_wavelet,
            self.build_pipeline_bandpower,
            self.build_pipeline_fourier,
            self.build_pipeline_autoregressive,
            self.build_pipeline_skewness_kurtosis,
            self.build_pipeline_autoencoder,
            self.build_pipeline_hjorth,
            self.build_pipeline_fractal,
            ##self.build_pipeline_combined_features
        ]

        for pipeline_fn in tqdm(pipelines, desc="Evaluating Pipelines"):
            print(f"Testing pipeline: {pipeline_fn.__name__}")
            self.build_evaluate_and_compare(pipeline_fn)
