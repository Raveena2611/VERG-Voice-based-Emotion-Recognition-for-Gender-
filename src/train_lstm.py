import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def build_lstm(input_shape, num_classes):
    model = Sequential()

    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    return model

def train_lstm(X_train, y_train, X_val, y_val, num_classes):
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_lstm(input_shape, num_classes)

    # class weights
    cw = compute_class_weight(class_weight="balanced",
                              classes=np.unique(y_train),
                              y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return model