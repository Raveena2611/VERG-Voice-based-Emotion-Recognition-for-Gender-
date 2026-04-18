from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=1200,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train, y_train)
    return rf