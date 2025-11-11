"""
Pomoćne funkcije za projekat
"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, model_name, path='results/models/'):
    """
    Čuvanje modela na disk
    
    Args:
        model: Model objekat
        model_name: Ime modela (npr. 'naive_bayes.pkl')
        path: Putanja do foldera
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, model_name)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model sačuvan: {filepath}")


def load_model(model_name, path='results/models/'):
    """
    Učitavanje modela sa diska
    
    Args:
        model_name: Ime modela
        path: Putanja do foldera
    
    Returns:
        model: Učitan model
    """
    filepath = os.path.join(path, model_name)
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model učitan: {filepath}")
    return model


def save_metrics(metrics, filename, path='results/metrics/'):
    """
    Čuvanje metrika u JSON format
    
    Args:
        metrics: Dictionary sa metrikama
        filename: Ime fajla
        path: Putanja do foldera
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    
    # Dodaj timestamp
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"Metrike sačuvane: {filepath}")


def save_figure(fig, filename, path='results/figures/', dpi=300):
    """
    Čuvanje grafika u visokoj rezoluciji
    
    Args:
        fig: Matplotlib figure objekat
        filename: Ime fajla (npr. 'confusion_matrix.png')
        path: Putanja do foldera
        dpi: Rezolucija
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Grafik sačuvan: {filepath}")


def load_twitter_data(filepath):
    """
    Učitavanje Twitter dataseta
    
    Args:
        filepath: Putanja do CSV fajla
    
    Returns:
        df: Pandas DataFrame
    """
    print(f"Učitavanje podataka iz: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Učitano {len(df)} redova")
    return df


def set_style():
    """
    Postavljanje stila za vizualizacije
    """
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def print_dataset_info(df):
    """
    Ispisuje osnovne informacije o datasetu
    
    Args:
        df: Pandas DataFrame
    """
    print("\n" + "="*50)
    print("INFORMACIJE O DATASETU")
    print("="*50)
    print(f"\nDimenzije: {df.shape}")
    print(f"\nKolone: {list(df.columns)}")
    print(f"\nTipovi podataka:")
    print(df.dtypes)
    print(f"\nNedostajuće vrednosti:")
    print(df.isnull().sum())
    print(f"\nPrvih 5 redova:")
    print(df.head())
    print("="*50 + "\n")