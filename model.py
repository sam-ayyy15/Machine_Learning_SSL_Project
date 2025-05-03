import numpy as np
import pandas as pd
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_preprocessed_data(input_dir='data'):
    """Load the preprocessed data"""
    X = pickle.load(open(f"{input_dir}/X_features.pkl", 'rb'))
    y = pickle.load(open(f"{input_dir}/y_labels.pkl", 'rb'))
    vectorizer = pickle.load(open(f"{input_dir}/count_vectorizer.pkl", 'rb'))
    
    print(f"Loaded preprocessed data. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, vectorizer

def split_data(X, y, test_size=0.1, train_size=0.25, random_state=42):
    """
    Split the data into training, test, and unlabeled sets
    Args:
        X: Features
        y: Labels
        test_size: Percentage of data for testing
        train_size: Percentage of data for initial training
        random_state: Random seed for reproducibility
    """
    # First split to get test data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Calculate the proportion of training data from the remaining data
    remaining_train_size = train_size / (1 - test_size)
    
    # Split again to get training and unlabeled data
    X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(
        X_temp, y_temp, 
        train_size=remaining_train_size, 
        random_state=random_state, 
        stratify=y_temp
    )
    
    print(f"Data split into: Train {X_train.shape[0]} samples, Test {X_test.shape[0]} samples, Unlabeled {X_unlabeled.shape[0]} samples")
    
    return X_train, X_test, X_unlabeled, y_train, y_test, y_unlabeled

def train_supervised_model(X_train, y_train, X_test, y_test):
    """Train a supervised SVM model"""
    print("Training supervised SVM model...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    cv_accuracy = np.mean(cv_scores)
    
    # Calculate training accuracy
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    print(f"Supervised Model - Test Accuracy: {accuracy:.4f}, CV Accuracy: {cv_accuracy:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    results = {
        'model': model,
        'test_accuracy': accuracy,
        'cv_accuracy': cv_accuracy,
        'train_accuracy': train_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return results

def self_training(X_train_init, y_train_init, X_unlabeled, X_test, y_test, probability_threshold=0.9):
    """
    Implement self-training semi-supervised learning
    Args:
        X_train_init: Initial training features
        y_train_init: Initial training labels
        X_unlabeled: Unlabeled features
        X_test: Test features
        y_test: Test labels
        probability_threshold: Threshold for pseudo-labeling
    """
    print(f"Starting self-training with probability threshold {probability_threshold}...")
    
    # Initialize variables
    X_train = X_train_init.copy()
    y_train = y_train_init.copy()
    X_remain = X_unlabeled.copy()
    
    # Convert to consistent data format
    # If any of the inputs are sparse, convert all to dense
    if hasattr(X_train, 'toarray') or hasattr(X_remain, 'toarray') or hasattr(X_test, 'toarray'):
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if hasattr(X_remain, 'toarray'):
            X_remain = X_remain.toarray()
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
    
    # Lists to track metrics over iterations
    metrics_per_iteration = []
    pseudo_labels_count = []
    
    # Start self-training iterations
    iteration = 0
    continue_training = True
    
    while continue_training and iteration < 30:  # Set a max iteration limit
        iteration += 1
        print(f"\nIteration {iteration}")
        
        # Train the model on the current labeled data
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on unlabeled data with probabilities
        if X_remain.shape[0] > 0:
            probabilities = model.predict_proba(X_remain)
            max_probs = np.max(probabilities, axis=1)
            
            # Find indices of high confidence predictions
            high_conf_indices = np.where(max_probs >= probability_threshold)[0]
            
            if len(high_conf_indices) > 0:
                # Get the predicted labels for high confidence samples
                pseudo_labels = model.predict(X_remain[high_conf_indices])
                
                # Add high confidence samples to training data
                X_train = np.vstack((X_train, X_remain[high_conf_indices]))
                y_train = np.append(y_train, pseudo_labels)
                
                # Remove labeled samples from unlabeled data
                mask = np.ones(X_remain.shape[0], dtype=bool)
                mask[high_conf_indices] = False
                X_remain = X_remain[mask]
                
                print(f"Added {len(high_conf_indices)} pseudo-labeled samples. Remaining unlabeled: {X_remain.shape[0]}")
                pseudo_labels_count.append(len(high_conf_indices))
            else:
                print("No high confidence predictions found. Stopping self-training.")
                continue_training = False
                pseudo_labels_count.append(0)
        else:
            print("No unlabeled data remaining. Stopping self-training.")
            continue_training = False
            pseudo_labels_count.append(0)
        
        # Evaluate on test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Calculate cross-validation score (if we have enough samples per class)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(10, np.min(np.bincount(y_train))))
            cv_accuracy = np.mean(cv_scores)
        except:
            cv_accuracy = np.nan
        
        # Calculate training accuracy
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"Iteration {iteration} - Test Accuracy: {accuracy:.4f}, CV Accuracy: {cv_accuracy:.4f}, Training Accuracy: {train_accuracy:.4f}")
        
        # Store metrics for this iteration
        metrics_per_iteration.append({
            'iteration': iteration,
            'test_accuracy': accuracy,
            'cv_accuracy': cv_accuracy,
            'train_accuracy': train_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_size': len(y_train)
        })
        
        # Stop if no more high confidence predictions or no more unlabeled data
        if not continue_training or X_remain.shape[0] == 0:
            break
    
    print(f"Self-training completed after {iteration} iterations.")
    
    # Final model
    final_model = SVC(kernel='linear', probability=True, random_state=42)
    final_model.fit(X_train, y_train)
    
    results = {
        'model': final_model,
        'metrics_per_iteration': metrics_per_iteration,
        'pseudo_labels_count': pseudo_labels_count,
        'final_training_size': len(y_train),
        'remaining_unlabeled': X_remain.shape[0]
    }
    
    return results

def run_experiments(X, y, train_sizes=[0.15, 0.20, 0.25, 0.80], test_size=0.1):
    """Run experiments with different training sizes"""
    all_results = {}
    
    for train_size in train_sizes:
        print(f"\n{'='*50}")
        print(f"Running experiment with {train_size*100}% training data")
        print(f"{'='*50}")
        
        if train_size == 0.80:  # Supervised learning
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            results = train_supervised_model(X_train, y_train, X_test, y_test)
            all_results[f'supervised_{train_size}'] = results
        else:  # Semi-supervised learning
            X_train, X_test, X_unlabeled, y_train, y_test, y_unlabeled = split_data(
                X, y, test_size=test_size, train_size=train_size
            )
            results = self_training(X_train, y_train, X_unlabeled, X_test, y_test)
            all_results[f'self_training_{train_size}'] = results
    
    return all_results

def save_results(all_results, output_dir='results'):
    """Save the experiment results"""
    os.makedirs(output_dir, exist_ok=True)
    
    for experiment_name, results in all_results.items():
        # Save the model
        if 'model' in results:
            pickle.dump(results['model'], open(f"{output_dir}/{experiment_name}_model.pkl", 'wb'))
        
        # Save metrics
        if 'metrics_per_iteration' in results:
            metrics_df = pd.DataFrame(results['metrics_per_iteration'])
            metrics_df.to_csv(f"{output_dir}/{experiment_name}_metrics.csv", index=False)
        
        # Save pseudo labels count
        if 'pseudo_labels_count' in results:
            np.save(f"{output_dir}/{experiment_name}_pseudo_labels_count.npy", results['pseudo_labels_count'])
    
    # Save a summary of results
    summary = {}
    for experiment_name, results in all_results.items():
        if 'metrics_per_iteration' in results:
            final_metrics = results['metrics_per_iteration'][-1]
            summary[experiment_name] = {
                'test_accuracy': final_metrics['test_accuracy'],
                'cv_accuracy': final_metrics['cv_accuracy'],
                'train_accuracy': final_metrics['train_accuracy'],
                'precision': final_metrics['precision'].tolist(),
                'recall': final_metrics['recall'].tolist(),
                'f1': final_metrics['f1'].tolist(),
                'final_training_size': results.get('final_training_size', 'N/A'),
                'remaining_unlabeled': results.get('remaining_unlabeled', 'N/A')
            }
        else:
            summary[experiment_name] = {
                'test_accuracy': results['test_accuracy'],
                'cv_accuracy': results['cv_accuracy'],
                'train_accuracy': results['train_accuracy'],
                'precision': results['precision'].tolist(),
                'recall': results['recall'].tolist(),
                'f1': results['f1'].tolist()
            }
    
    with open(f"{output_dir}/summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    # Load preprocessed data
    X, y, vectorizer = load_preprocessed_data()
    
    # Run experiments
    all_results = run_experiments(X, y)
    
    # Save results
    save_results(all_results)