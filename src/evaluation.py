"""
Funkcije za evaluaciju modela
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluacija modela - sve metrike
    
    Args:
        y_true: Prave labele
        y_pred: Predviđene labele
        model_name: Ime modela
        
    Returns:
        Dictionary sa metrikama
    """
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def print_evaluation_report(y_true, y_pred, model_name="Model"):
    """
    Ispisuje detaljnu evaluaciju modela
    
    Args:
        y_true: Prave labele
        y_pred: Predviđene labele
        model_name: Ime modela
    """
    metrics = evaluate_model(y_true, y_pred, model_name)
    
    print("\n" + "="*60)
    print(f"EVALUACIJA MODELA: {model_name}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("="*60 + "\n")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, model_name="Model"):
    """
    Plotuje confusion matrix
    
    Args:
        y_true: Prave labele
        y_pred: Predviđene labele
        labels: Lista labela (npr. ['negative', 'neutral', 'positive'])
        model_name: Ime modela
        
    Returns:
        Matplotlib figure objekat
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predviđeno', fontsize=12)
    ax.set_ylabel('Stvarno', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_list):
    """
    Plotuje poređenje metrika između više modela
    
    Args:
        metrics_list: Lista dictionary-a sa metrikama
        
    Returns:
        Matplotlib figure objekat
    """
    df = pd.DataFrame(metrics_list)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.2
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title(), color=color)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Poređenje Performansi Modela', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['model_name'])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_results_table(metrics_list):
    """
    Kreira tabelu sa rezultatima
    
    Args:
        metrics_list: Lista dictionary-a sa metrikama
        
    Returns:
        Pandas DataFrame
    """
    df = pd.DataFrame(metrics_list)
    
    # Formatiranje
    for col in ['accuracy', 'precision', 'recall', 'f1_score']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    return df


def plot_sentiment_distribution(y_true, y_pred, labels=None, model_name="Model"):
    """
    Plotuje distribuciju sentiment-a (prave vs predviđene)
    
    Args:
        y_true: Prave labele
        y_pred: Predviđene labele
        labels: Lista labela
        model_name: Ime modela
        
    Returns:
        Matplotlib figure objekat
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prave labele
    unique, counts = np.unique(y_true, return_counts=True)
    axes[0].bar(range(len(unique)), counts, color='#3498db')
    axes[0].set_xlabel('Sentiment', fontsize=12)
    axes[0].set_ylabel('Broj primeraka', fontsize=12)
    axes[0].set_title('Stvarna Distribucija', fontsize=12, fontweight='bold')
    if labels:
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels)
    
    # Predviđene labele
    unique, counts = np.unique(y_pred, return_counts=True)
    axes[1].bar(range(len(unique)), counts, color='#e74c3c')
    axes[1].set_xlabel('Sentiment', fontsize=12)
    axes[1].set_ylabel('Broj primeraka', fontsize=12)
    axes[1].set_title('Predviđena Distribucija', fontsize=12, fontweight='bold')
    if labels:
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
    
    fig.suptitle(f'Distribucija Sentimenta - {model_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


# Primer korišćenja
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST EVALUATION FUNKCIJA")
    print("="*60)
    
    # Dummy podaci
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 0, 2])
    
    # Test
    metrics = print_evaluation_report(y_true, y_pred, "Test Model")
    
    print("\n✅ Evaluation testovi prošli uspešno!")
    print("="*60)