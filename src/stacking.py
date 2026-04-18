import numpy as np
from sklearn.linear_model import LogisticRegression

def train_stacking(rf_model, lstm_model, X_rf_val, X_seq_val, y_val):
    
    rf_proba = rf_model.predict_proba(X_rf_val)
    lstm_proba = lstm_model.predict(X_seq_val, verbose=0)

    stack_X = np.hstack([rf_proba, lstm_proba])

    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(stack_X, y_val)

    return meta_model

def stacking_predict(rf_model, lstm_model, meta_model, X_rf_test, X_seq_test):
    
    rf_proba = rf_model.predict_proba(X_rf_test)
    lstm_proba = lstm_model.predict(X_seq_test, verbose=0)

    stack_X = np.hstack([rf_proba, lstm_proba])

    return meta_model.predict(stack_X)