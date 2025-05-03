import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_results(results_dir='results'):
    """Load experiment results"""
    # Load the summary file
    with open(f"{results_dir}/summary.pkl", 'rb') as f:
        summary = pickle.load(f)
    
    # Load label mapping
    with open("data/label_mapping.pkl", 'rb') as f:
        label_mapping = pickle.load(f)
    
    # Reverse the label mapping
    rev_label_mapping = {v: k for k, v in label_mapping.items()}
    
    return summary, rev_label_mapping

def plot_iteration_metrics(results_dir='results'):
    """Plot metrics across iterations for self-training experiments"""
    files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    
    plt.figure(figsize=(15, 10))
    
    for i, file in enumerate(files):
        if 'self_training' in file:
            # Extract training percentage from filename
            train_pct = file.split('_')[2].split('.')[0]
            
            # Load metrics
            metrics_df = pd.read_csv(f"{results_dir}/{file}")
            
            # Plot test accuracy
            plt.subplot(2, 2, 1)
            plt.plot(metrics_df['iteration'], metrics_df['test_accuracy'], 
                     marker='o', label=f"{train_pct}% Training")
            plt.xlabel('Iteration')
            plt.ylabel('Test Accuracy')
            plt.title('Test Accuracy across Iterations')
            plt.grid(True)
            plt.legend()
            
            # Plot training size
            plt.subplot(2, 2, 2)
            plt.plot(metrics_df['iteration'], metrics_df['training_size'], 
                     marker='o', label=f"{train_pct}% Training")
            plt.xlabel('Iteration')
            plt.ylabel('Training Set Size')
            plt.title('Growth of Training Set')
            plt.grid(True)
            plt.legend()
            
            # Plot precision for each class (using first class as example)
            plt.subplot(2, 2, 3)
            precision_cols = [col for col in metrics_df.columns if col.startswith('precision')]
            if len(precision_cols) > 0:
                for j, col in enumerate(['precision[0]', 'precision[1]', 'precision[2]']):
                    if col in metrics_df.columns:
                        plt.plot(metrics_df['iteration'], metrics_df[col], 
                                marker='o', label=f"Class {j} ({train_pct}%)")
            plt.xlabel('Iteration')
            plt.ylabel('Precision')
            plt.title('Precision by Class')
            plt.grid(True)
            plt.legend()
            
            # Plot recall for each class
            plt.subplot(2, 2, 4)
            recall_cols = [col for col in metrics_df.columns if col.startswith('recall')]
            if len(recall_cols) > 0:
                for j, col in enumerate(['recall[0]', 'recall[1]', 'recall[2]']):
                    if col in metrics_df.columns:
                        plt.plot(metrics_df['iteration'], metrics_df[col], 
                                marker='o', label=f"Class {j} ({train_pct}%)")
            plt.xlabel('Iteration')
            plt.ylabel('Recall')
            plt.title('Recall by Class')
            plt.grid(True)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/iteration_metrics.png")
    plt.close()
    
    print(f"Iteration metrics plot saved to {results_dir}/iteration_metrics.png")

def plot_pseudo_label_counts(results_dir='results'):
    """Plot the number of pseudo-labels added in each iteration"""
    plt.figure(figsize=(10, 6))
    
    for file in os.listdir(results_dir):
        if file.endswith('_pseudo_labels_count.npy'):
            # Extract training percentage from filename
            train_pct = file.split('_')[2].split('.')[0]
            
            # Load pseudo label counts
            counts = np.load(f"{results_dir}/{file}")
            
            # Plot counts
            plt.plot(range(1, len(counts)+1), counts, marker='o', label=f"{train_pct}% Training")
    
    plt.xlabel('Iteration')
    plt.ylabel('Number of Pseudo-Labels Added')
    plt.title('Pseudo-Labels Added per Iteration')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{results_dir}/pseudo_labels_count.png")
    plt.close()
    
    print(f"Pseudo-label counts plot saved to {results_dir}/pseudo_labels_count.png")

def compare_models(summary, rev_label_mapping, results_dir='results'):
    """Compare the performance of different models"""
    # Extract metrics
    experiments = []
    accuracies = []
    training_sizes = []
    
    for exp_name, metrics in summary.items():
        experiments.append(exp_name)
        accuracies.append(metrics['test_accuracy'])
        
        # Training size is only available for self-training experiments
        if 'final_training_size' in metrics and metrics['final_training_size'] != 'N/A':
            training_sizes.append(metrics['final_training_size'])
        else:
            # For supervised model, we need to estimate based on the training percentage
            if 'supervised_0.8' in exp_name:
                training_sizes.append(4511 * 0.8)  # 80% of total dataset
    
    # Create a comparison dataframe
    comparison_df = pd.DataFrame({
        'Experiment': experiments,
        'Test Accuracy': accuracies,
        'Training Size': training_sizes
    })
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
    
    # Display the comparison
    print("Model Comparison:")
    print(comparison_df)
    
    # Plot the comparison
    plt.figure(figsize=(12, 6))
    
    # Bar chart of accuracies
    plt.subplot(1, 2, 1)
    plt.bar(comparison_df['Experiment'], comparison_df['Test Accuracy'])
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy by Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Plot accuracy vs training size
    plt.subplot(1, 2, 2)
    plt.scatter(comparison_df['Training Size'], comparison_df['Test Accuracy'])
    
    # Add labels for each point
    for i, txt in enumerate(comparison_df['Experiment']):
        plt.annotate(txt, 
                    (comparison_df['Training Size'].iloc[i], comparison_df['Test Accuracy'].iloc[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.xlabel('Training Size')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Training Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/model_comparison.png")
    plt.close()
    
    print(f"Model comparison plot saved to {results_dir}/model_comparison.png")
    
    # Save the comparison to CSV
    comparison_df.to_csv(f"{results_dir}/model_comparison.csv", index=False)
    print(f"Model comparison saved to {results_dir}/model_comparison.csv")
    
    return comparison_df

if __name__ == "__main__":
    print("Analyzing results...")
    
    # Load results
    summary, rev_label_mapping = load_results()
    
    # Plot iteration metrics
    plot_iteration_metrics()
    
    # Plot pseudo label counts
    plot_pseudo_label_counts()
    
    # Compare models
    comparison_df = compare_models(summary, rev_label_mapping)
    
    print("Analysis complete!")