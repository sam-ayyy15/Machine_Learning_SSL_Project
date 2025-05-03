import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

def load_data(data_dir='data', results_dir='results'):
    """Load all necessary data for visualizations"""
    print("Loading data for visualization...")
    
    # Load original dataset
    df = pd.read_csv(f"{data_dir}/synthetic_asian_hate_tweets.csv")
    
    # Load preprocessed data
    X = pickle.load(open(f"{data_dir}/X_features.pkl", 'rb'))
    y = pickle.load(open(f"{data_dir}/y_labels.pkl", 'rb'))
    vectorizer = pickle.load(open(f"{data_dir}/count_vectorizer.pkl", 'rb'))
    
    # Load label mapping
    with open(f"{data_dir}/label_mapping.pkl", 'rb') as f:
        label_mapping = pickle.load(f)
    
    # Reverse the label mapping for easier interpretation
    rev_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Load model results
    try:
        with open(f"{results_dir}/summary.pkl", 'rb') as f:
            summary = pickle.load(f)
    except FileNotFoundError:
        summary = None
        print("Warning: Results summary not found. Run model experiments first.")
    
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    return df, X, y, vectorizer, label_mapping, rev_label_mapping, summary

def visualize_dataset_stats(df, output_dir='visualizations'):
    """Visualize basic dataset statistics"""
    print("Generating dataset statistics visualizations...")
    
    # 1. Class distribution
    plt.figure(figsize=(12, 6))
    
    # Plot label distribution as pie chart
    plt.subplot(1, 2, 1)
    label_counts = df['label'].value_counts()
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette('Set2', n_colors=len(label_counts)))
    plt.axis('equal')
    plt.title('Tweet Classification Distribution')
    
    # Plot label distribution as bar chart
    plt.subplot(1, 2, 2)
    sns.countplot(x='label', data=df, palette='Set2')
    plt.title('Count of Tweets by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.close()
    
    # 2. Tweet length distribution by category
    plt.figure(figsize=(14, 7))
    
    # Add tweet length to dataframe
    df['tweet_length'] = df['text'].apply(len)
    
    # Plot tweet length distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='tweet_length', hue='label', kde=True, palette='Set2')
    plt.title('Tweet Length Distribution by Category')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Count')
    
    # Box plot of tweet lengths
    plt.subplot(1, 2, 2)
    sns.boxplot(x='label', y='tweet_length', data=df, palette='Set2')
    plt.title('Tweet Length Comparison by Category')
    plt.xlabel('Category')
    plt.ylabel('Tweet Length (characters)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tweet_length_distribution.png")
    plt.close()
    
    # 3. Tweet posting over time
    plt.figure(figsize=(14, 7))
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Plot tweets over time by category
    monthly_counts = df.groupby(['month', 'label']).size().unstack()
    monthly_counts.plot(kind='bar', figsize=(14, 7), width=0.8)
    plt.title('Tweet Distribution Over Time by Category')
    plt.xlabel('Month')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Category')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tweets_over_time.png")
    plt.close()
    
    # 4. Word count distribution by category
    plt.figure(figsize=(14, 7))
    
    # Add word count to dataframe
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    # Plot word count distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='word_count', hue='label', kde=True, palette='Set2')
    plt.title('Word Count Distribution by Category')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    # Box plot of word counts
    plt.subplot(1, 2, 2)
    sns.boxplot(x='label', y='word_count', data=df, palette='Set2')
    plt.title('Word Count Comparison by Category')
    plt.xlabel('Category')
    plt.ylabel('Word Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/word_count_distribution.png")
    plt.close()
    
    return df

def generate_wordclouds(df, output_dir='visualizations'):
    """Generate wordclouds for each category"""
    print("Generating wordclouds...")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for each category
    colors = {
        'Hate': 'Reds',
        'Neutral': 'Blues',
        'Counter hate': 'Greens'
    }
    
    # Additional stopwords
    additional_stops = {'will', 'covid', 'covid19', 'coronavirus'}
    custom_stopwords = STOPWORDS.union(additional_stops)
    
    # Generate wordcloud for each category
    for i, category in enumerate(['Hate', 'Neutral', 'Counter hate']):
        # Combine all text in this category
        text = ' '.join(df[df['label'] == category]['text'])
        
        # Create and generate the wordcloud
        wc = WordCloud(
            width=800, height=400,
            max_words=100,
            background_color='white',
            colormap=colors[category],
            stopwords=custom_stopwords,
            random_state=42
        ).generate(text)
        
        # Display the wordcloud on the corresponding subplot
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f'{category} Speech Wordcloud')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wordclouds.png")
    plt.close()
    
    # Also generate comparative barcharts of top words
    plt.figure(figsize=(15, 20))
    
    for i, category in enumerate(['Hate', 'Neutral', 'Counter hate']):
        plt.subplot(3, 1, i+1)
        
        # Get text from this category
        texts = df[df['label'] == category]['text'].tolist()
        
        # Count words
        word_counts = Counter()
        for text in texts:
            # Skip stopwords
            words = [word.lower() for word in text.split() if word.lower() not in custom_stopwords]
            word_counts.update(words)
        
        # Get top 15 words
        top_words = dict(word_counts.most_common(15))
        
        # Create bar chart
        sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), 
                   color=sns.color_palette(colors[category])[3],
                   orient='h')
        
        plt.title(f'Top 15 Words in {category} Tweets')
        plt.xlabel('Frequency')
        plt.ylabel('Word')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_words_by_category.png")
    plt.close()

def visualize_feature_space(X, y, rev_label_mapping, output_dir='visualizations'):
    """Visualize the feature space using dimensionality reduction"""
    print("Visualizing feature space...")
    
    # Convert sparse matrix to dense if needed
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # 1. PCA visualization
    plt.figure(figsize=(15, 6))
    
    # Apply PCA
    plt.subplot(1, 2, 1)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    
    # Get human-readable labels
    y_labels = [rev_label_mapping[label] for label in y]
    
    # Create scatter plot
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette='Set2', alpha=0.7)
    plt.title('PCA Visualization of Tweet Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Category')
    
    # 2. t-SNE visualization 
    plt.subplot(1, 2, 2)
    
    # Use a sample if the dataset is large (t-SNE is computationally intensive)
    sample_size = min(3000, X_dense.shape[0])
    indices = np.random.choice(X_dense.shape[0], sample_size, replace=False)
    
    # Apply t-SNE to the sample
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_dense[indices])
    
    # Create scatter plot
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], 
                   hue=[rev_label_mapping[y[i]] for i in indices], 
                   palette='Set2', alpha=0.7)
    plt.title('t-SNE Visualization of Tweet Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Category')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_space_visualization.png")
    plt.close()

def visualize_feature_importance(vectorizer, model_results, output_dir='visualizations'):
    """Visualize the most important features (words) for classification"""
    print("Visualizing feature importance...")
    
    # Loop through available models
    for model_name, results in model_results.items():
        if 'model' in results:
            model = results['model']
            
            # Check if model has coef_ attribute (linear models)
            if hasattr(model, 'coef_'):
                plt.figure(figsize=(16, 12))
                
                # Get feature names
                feature_names = np.array(vectorizer.get_feature_names_out())
                
                # Number of classes
                n_classes = model.coef_.shape[0]
                
                # Check if multiclass or binary
                for i in range(n_classes):
                    plt.subplot(n_classes, 1, i+1)
                    
                    # Get coefficients for this class and convert to dense if needed
                    if hasattr(model.coef_, 'toarray'):
                        coefs = model.coef_[i].toarray().flatten()
                    else:
                        coefs = model.coef_[i]
                    
                    # Get top positive and negative coefficients
                    top_positive_coefs = np.argsort(coefs)[-15:]
                    top_negative_coefs = np.argsort(coefs)[:15]
                    top_coefficients = np.hstack([top_negative_coefs, top_positive_coefs])
                    
                    # Get coefficient values for selected features
                    coef_values = coefs[top_coefficients]
                    
                    # Plot horizontal bar chart
                    colors = ['red' if x < 0 else 'green' for x in coef_values]
                    plt.barh(feature_names[top_coefficients], coef_values, color=colors)
                    plt.title(f"Class {i} Top Features")
                    plt.xlabel("Coefficient Value")
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{model_name}_feature_importance.png")
                plt.close()
                
                print(f"Feature importance visualization saved for {model_name}")

def visualize_confusion_matrices(summary, results_dir='results', output_dir='visualizations'):
    """Visualize confusion matrices for all models"""
    print("Visualizing confusion matrices...")
    
    # Iterate through models
    for exp_name in summary.keys():
        # Try to load test predictions
        try:
            # For self-training models
            if 'self_training' in exp_name:
                # Load the model
                model = pickle.load(open(f"{results_dir}/{exp_name}_model.pkl", 'rb'))
                
                # Load test data
                X_test = pickle.load(open(f"{results_dir}/{exp_name}_X_test.pkl", 'rb'))
                y_test = pickle.load(open(f"{results_dir}/{exp_name}_y_test.pkl", 'rb'))
                
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Visualize confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {exp_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(f"{output_dir}/{exp_name}_confusion_matrix.png")
                plt.close()
                
                print(f"Confusion matrix saved for {exp_name}")
                
        except FileNotFoundError:
            # If test data isn't available, skip
            print(f"Test data not found for {exp_name}, skipping confusion matrix")

def visualize_learning_curves(results_dir='results', output_dir='visualizations'):
    """Visualize learning curves for self-training models"""
    print("Visualizing learning curves...")
    
    # Find all metrics files
    metrics_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    
    if not metrics_files:
        print("No metrics files found. Run model experiments first.")
        return
    
    # Plot accuracy curves
    plt.figure(figsize=(12, 8))
    
    for file in metrics_files:
        # Extract experiment name
        exp_name = file.replace('_metrics.csv', '')
        
        # Load metrics
        metrics_df = pd.read_csv(f"{results_dir}/{file}")
        
        # Plot test accuracy over iterations
        plt.plot(metrics_df['iteration'], metrics_df['test_accuracy'], 
                 marker='o', label=exp_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Over Self-Training Iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/learning_curves.png")
    plt.close()
    
    # Plot training size growth
    plt.figure(figsize=(12, 8))
    
    for file in metrics_files:
        # Extract experiment name
        exp_name = file.replace('_metrics.csv', '')
        
        # Load metrics
        metrics_df = pd.read_csv(f"{results_dir}/{file}")
        
        # Plot training size over iterations if available
        if 'training_size' in metrics_df.columns:
            plt.plot(metrics_df['iteration'], metrics_df['training_size'], 
                     marker='o', label=exp_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Training Set Size')
    plt.title('Growth of Training Set Over Self-Training Iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/training_size_growth.png")
    plt.close()

def visualize_comparative_metrics(summary, output_dir='visualizations'):
    """Visualize comparative metrics across all models"""
    print("Visualizing comparative metrics...")
    
    if not summary:
        print("No summary data available. Run model experiments first.")
        return
    
    # Extract metrics for comparison
    models = []
    test_acc = []
    cv_acc = []
    train_acc = []
    train_sizes = []
    
    for model_name, metrics in summary.items():
        models.append(model_name)
        test_acc.append(metrics['test_accuracy'])
        cv_acc.append(metrics['cv_accuracy'])
        train_acc.append(metrics['train_accuracy'])
        
        # Training size might not be available for all models
        if 'final_training_size' in metrics and metrics['final_training_size'] != 'N/A':
            train_sizes.append(metrics['final_training_size'])
        else:
            train_sizes.append(None)
    
    # Create DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Model': models,
        'Test Accuracy': test_acc,
        'CV Accuracy': cv_acc,
        'Training Accuracy': train_acc,
        'Training Size': train_sizes
    })
    
    # Sort by test accuracy
    metrics_df = metrics_df.sort_values('Test Accuracy', ascending=False)
    
    # Plot bar chart of accuracies
    plt.figure(figsize=(14, 8))
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(metrics_df))
    width = 0.25
    
    # Create the bars
    plt.bar(x - width, metrics_df['Test Accuracy'], width, label='Test Accuracy', color='#2196F3')
    plt.bar(x, metrics_df['CV Accuracy'], width, label='CV Accuracy', color='#4CAF50')
    plt.bar(x + width, metrics_df['Training Accuracy'], width, label='Training Accuracy', color='#FF9800')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.xticks(x, metrics_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/model_accuracy_comparison.png")
    plt.close()
    
    # Plot accuracy vs training size
    plt.figure(figsize=(12, 8))
    
    # Filter out models without training size
    size_df = metrics_df.dropna(subset=['Training Size'])
    
    # Create scatter plot
    plt.scatter(size_df['Training Size'], size_df['Test Accuracy'], s=100, color='#2196F3')
    
    # Add labels for each point
    for i, model in enumerate(size_df['Model']):
        plt.annotate(model, 
                    (size_df['Training Size'].iloc[i], size_df['Test Accuracy'].iloc[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Add best fit line
    if len(size_df) > 1:
        z = np.polyfit(size_df['Training Size'], size_df['Test Accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(size_df['Training Size'], p(size_df['Training Size']), "r--", alpha=0.8)
    
    plt.xlabel('Training Size')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Training Size')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"{output_dir}/accuracy_vs_training_size.png")
    plt.close()

def visualize_class_specific_metrics(summary, output_dir='visualizations'):
    """Visualize precision, recall, and F1 for each class across models"""
    print("Visualizing class-specific metrics...")
    
    if not summary:
        print("No summary data available. Run model experiments first.")
        return
    
    # Check if we have class-specific metrics
    first_model = list(summary.keys())[0]
    if 'precision' not in summary[first_model]:
        print("Class-specific metrics not available in summary.")
        return
    
    # Create dataframes for class metrics
    precision_df = pd.DataFrame(index=summary.keys())
    recall_df = pd.DataFrame(index=summary.keys())
    f1_df = pd.DataFrame(index=summary.keys())
    
    # Extract metrics for each class
    for model_name, metrics in summary.items():
        # Check for precision data
        if 'precision' in metrics:
            for i, val in enumerate(metrics['precision']):
                precision_df.loc[model_name, f'Class {i}'] = val
        
        # Check for recall data
        if 'recall' in metrics:
            for i, val in enumerate(metrics['recall']):
                recall_df.loc[model_name, f'Class {i}'] = val
        
        # Check for F1 data
        if 'f1' in metrics:
            for i, val in enumerate(metrics['f1']):
                f1_df.loc[model_name, f'Class {i}'] = val
    
    # Plot class-specific metrics
    metrics = [
        ('Precision', precision_df),
        ('Recall', recall_df),
        ('F1 Score', f1_df)
    ]
    
    for metric_name, df in metrics:
        plt.figure(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': metric_name})
        
        plt.title(f'{metric_name} by Class and Model')
        plt.ylabel('Model')
        plt.xlabel('Class')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{output_dir}/{metric_name.lower()}_by_class.png")
        plt.close()
    
    # Create a radar chart for comprehensive comparison
    categories = df.columns
    n_cats = len(categories)
    
    # Select top 5 models based on test accuracy
    if len(summary) > 5:
        sorted_models = sorted(summary.keys(), 
                            key=lambda x: summary[x]['test_accuracy'], 
                            reverse=True)[:5]
    else:
        sorted_models = list(summary.keys())
    
    # Create radar chart
    fig = plt.figure(figsize=(15, 12))
    
    for i, metric_name in enumerate(['Precision', 'Recall', 'F1 Score']):
        metric_df = metrics[i][1]
        
        ax = fig.add_subplot(2, 2, i+1, polar=True)
        
        # Angles for each category
        angles = [n/n_cats * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for j, model in enumerate(sorted_models):
            values = metric_df.loc[model].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        ax.set_title(f'{metric_name} by Class for Top Models')
        ax.grid(True)
    
    # Add legend in the fourth subplot position
    ax = fig.add_subplot(2, 2, 4)
    ax.axis('off')
    ax.legend(sorted_models, loc="center")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_metrics_radar.png")
    plt.close()

def run_all_visualizations():
    """Run all visualization functions"""
    print("Running all visualizations...")
    
    # Load data
    df, X, y, vectorizer, label_mapping, rev_label_mapping, summary = load_data()
    
    # Run visualizations
    df = visualize_dataset_stats(df)
    generate_wordclouds(df)
    visualize_feature_space(X, y, rev_label_mapping)
    
    if summary:
        # Find available models
        model_results = {}
        for exp_name in summary.keys():
            try:
                model = pickle.load(open(f"results/{exp_name}_model.pkl", 'rb'))
                model_results[exp_name] = {'model': model}
            except FileNotFoundError:
                print(f"Model file not found for {exp_name}")
        
        # Visualize feature importance for available models
        visualize_feature_importance(vectorizer, model_results)
        
        # Visualize confusion matrices
        visualize_confusion_matrices(summary)
        
        # Visualize learning curves
        visualize_learning_curves()
        
        # Visualize comparative metrics
        visualize_comparative_metrics(summary)
        
        # Visualize class-specific metrics
        visualize_class_specific_metrics(summary)
    
    print("All visualizations complete!")

if __name__ == "__main__":
    run_all_visualizations()