    def extract_features(self, modality=None):
        '''
        Extracting features from the processed and epoched EEG data.
        Input: -
        Output: -
        '''
        if modality == 'bp':
            ### Feature 1: Bandpower ###
            # find optimal frequency bands
            freq_bands = [(3, 5),  # lower and upper freq-band limits
                          (10, 15),]
            # cut epochs dependend on frequency bands
            freq_epochs = [self.epochs.filter(l_freq=band[0], h_freq=band[1], method='iir') for band in freq_bands]
            # square the time-data and calculate the average
            # epoch.get_data() returns a numpy arrray of shape (num_epochs, num_channels, num_timepoints)
            bp_features = array([mean(epoch.get_data(copy=False)**2, axis=2) for epoch in freq_epochs])
            bp_features = reshape(bp_features, newshape=(bp_features.shape[1], -1))
            ground_truth = self.epochs.events[:, 2]
            
            # prepare Feature Variables Bandpower
            self.featureX_train = bp_features
            self.featureY_train = ground_truth
            self.featureX_test = bp_features
            self.featureY_test = ground_truth
        
        elif modality == 'csp':
            ### Feature 2: CSP ###
            # split data in train and test
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.epochs.get_data(copy=False),
                                                                                    self.epochs.events[:, 2],
                                                                                    train_size=0.7)
            # CSP with regularization 
            csp = RegularizedCSP(n_components=4 , reg=1e-8, log=True, norm_trace=False)  
            csp.fit(self.x_train, self.y_train)
            
            
            x_train_csp = csp.transform(self.x_train)
            reshape(x_train_csp, newshape=(x_train_csp.shape[1], -1))
            x_test_csp = csp.transform(self.x_test)
            reshape(x_test_csp, newshape=(x_test_csp.shape[1], -1))
            
            # prepare Feature Variables CSP
            self.featureX_train = x_train_csp
            self.featureY_train = self.y_train
            self.featureX_test = x_test_csp
            self.featureY_test = self.y_test
        else:
            raise ValueError('Please use a modality: f.e. bp or csp.')