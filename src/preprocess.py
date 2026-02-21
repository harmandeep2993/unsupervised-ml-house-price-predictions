import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler 

class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()      # work around means and std.
        self.robust = RobustScaler()        # work with median values. 
    
    def transform(self, X_train, X_test, type:):

        if self.
    
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled
    