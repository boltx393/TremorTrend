import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape), 
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate_models(X_train, X_test, X_train_lstm, X_test_lstm, y_train, y_test):
    results = []
    
    # Traditional ML models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    for name, model in models.items():
        print(f"\nTraining {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results.append(evaluate_model(y_test, y_pred, y_pred_proba, name))

    print("\nTraining LSTM model")
    input_shape = (1, X_train.shape[1])  
    lstm_model = create_lstm_model(input_shape)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min'
    )
    
    history = lstm_model.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate LSTM
    y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
    y_pred_proba_lstm = lstm_model.predict(X_test_lstm)
    results.append(evaluate_model(y_test, y_pred_lstm, y_pred_proba_lstm, 'LSTM'))
    
    return pd.DataFrame(results), history

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Calculate and return model performance metrics"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def preprocess_data(data):
    """Preprocess the data"""
    # Separate features and target
    X = data.drop('Landslide', axis=1)
    y = data['Landslide']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Reshape data for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    return X_train_scaled, X_test_scaled, X_train_lstm, X_test_lstm, y_train, y_test

def plot_correlations(data):
    """Plot correlation matrix"""
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_results(results, history):
    """Plot model comparisons and LSTM training history"""
    # Plot model comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    plt.figure(figsize=(15, 8))
    x = np.arange(len(results['Model']))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, results[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*2, results['Model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot LSTM training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main(data_path):
    """Main execution function"""
    data = pd.read_csv(data_path)
    print("Dataset Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Plot correlations
    plot_correlations(data)
    
    # Preprocess data
    X_train_scaled, X_test_scaled, X_train_lstm, X_test_lstm, y_train, y_test = preprocess_data(data)
    
    # Train and evaluate models
    results, history = train_and_evaluate_models(
        X_train_scaled, X_test_scaled,
        X_train_lstm, X_test_lstm,
        y_train, y_test
    )
    
    # Display results
    print("\nModel Performance Comparison:")
    print(results.round(4))
    
    # Plot results
    plot_results(results, history)

if __name__ == "__main__":
    data_path = "dataset.csv" 
    main(data_path)