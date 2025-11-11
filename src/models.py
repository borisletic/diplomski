"""
Implementacije mašinskog učenja modela
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class NaiveBayesModel:
    """
    Naive Bayes model sa TF-IDF vektorizacijom
    """
    
    def __init__(self, max_features=5000):
        """
        Args:
            max_features: Maksimalan broj feature-a za TF-IDF
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features)),
            ('classifier', MultinomialNB())
        ])
        self.is_trained = False
    
    
    def train(self, X_train, y_train):
        """
        Treniranje modela
        
        Args:
            X_train: Trening tekstovi
            y_train: Trening labele
        """
        print("Treniranje Naive Bayes modela...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print("Treniranje završeno!")
    
    
    def predict(self, X_test):
        """
        Predikcija
        
        Args:
            X_test: Test tekstovi
            
        Returns:
            Predviđene labele
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran! Pozovi train() metodu.")
        
        return self.pipeline.predict(X_test)
    
    
    def predict_proba(self, X_test):
        """
        Predikcija verovatnoća
        
        Args:
            X_test: Test tekstovi
            
        Returns:
            Verovatnoće za svaku klasu
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran! Pozovi train() metodu.")
        
        return self.pipeline.predict_proba(X_test)


class SVMModel:
    """
    SVM model sa TF-IDF vektorizacijom
    """
    
    def __init__(self, max_features=5000, kernel='rbf', C=1.0, gamma='scale'):
        """
        Args:
            max_features: Maksimalan broj feature-a za TF-IDF
            kernel: Kernel funkcija ('linear', 'rbf', 'poly')
            C: Regularizacioni parametar
            gamma: Kernel koeficijent
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features)),
            ('classifier', SVC(kernel=kernel, C=C, gamma=gamma, probability=True))
        ])
        self.is_trained = False
    
    
    def train(self, X_train, y_train):
        """
        Treniranje modela
        
        Args:
            X_train: Trening tekstovi
            y_train: Trening labele
        """
        print("Treniranje SVM modela...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print("Treniranje završeno!")
    
    
    def predict(self, X_test):
        """
        Predikcija
        
        Args:
            X_test: Test tekstovi
            
        Returns:
            Predviđene labele
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran! Pozovi train() metodu.")
        
        return self.pipeline.predict(X_test)
    
    
    def predict_proba(self, X_test):
        """
        Predikcija verovatnoća
        
        Args:
            X_test: Test tekstovi
            
        Returns:
            Verovatnoće za svaku klasu
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran! Pozovi train() metodu.")
        
        return self.pipeline.predict_proba(X_test)
    
    
    def grid_search(self, X_train, y_train, param_grid=None, cv=5):
        """
        Grid search za optimizaciju hiperparametara
        
        Args:
            X_train: Trening tekstovi
            y_train: Trening labele
            param_grid: Dictionary sa parametrima za pretragu
            cv: Broj fold-ova za cross-validation
        """
        if param_grid is None:
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__kernel': ['rbf', 'linear']
            }
        
        print("Pokretanje Grid Search...")
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=cv, 
                                    scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\nNajbolji parametri: {grid_search.best_params_}")
        print(f"Najbolji score: {grid_search.best_score_:.4f}")
        
        self.pipeline = grid_search.best_estimator_
        self.is_trained = True
        
        return grid_search.best_params_


# Primer korišćenja
if __name__ == "__main__":
    # Dummy podaci za test
    X_train = ["I love Bitcoin!", "Bitcoin is bad", "Not sure about crypto"]
    y_train = [2, 0, 1]  # 0: negative, 1: neutral, 2: positive
    X_test = ["Bitcoin rocks!", "I hate crypto"]
    
    print("\n" + "="*60)
    print("TEST MODELA")
    print("="*60)
    
    # Naive Bayes
    print("\n--- NAIVE BAYES ---")
    nb_model = NaiveBayesModel()
    nb_model.train(X_train, y_train)
    predictions = nb_model.predict(X_test)
    print(f"Predictions: {predictions}")
    
    # SVM
    print("\n--- SVM ---")
    svm_model = SVMModel(kernel='linear')
    svm_model.train(X_train, y_train)
    predictions = svm_model.predict(X_test)
    print(f"Predictions: {predictions}")
    
    print("\n" + "="*60)
    print("✅ Testovi prošli uspešno!")
    print("="*60)