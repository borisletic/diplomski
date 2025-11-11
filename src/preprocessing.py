"""
Funkcije za pretprocesiranje tekstualnih podataka
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download potrebnih NLTK resursa (pokrenuti jednom)
def download_nltk_resources():
    """
    Preuzimanje NLTK resursa
    """
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK resursi preuzeti uspešno")
    except Exception as e:
        print(f"⚠️  Greška pri preuzimanju NLTK resursa: {e}")


class TextPreprocessor:
    """
    Klasa za pretprocesiranje tekstualnih podataka
    """
    
    def __init__(self, lowercase=True, remove_urls=True, remove_mentions=True, 
                 remove_hashtags=True, remove_numbers=True, remove_punctuation=True,
                 remove_stopwords=True, stemming=False, lemmatization=True):
        """
        Args:
            lowercase: Pretvaranje u mala slova
            remove_urls: Uklanjanje URL-ova
            remove_mentions: Uklanjanje @mentions
            remove_hashtags: Uklanjanje #hashtags
            remove_numbers: Uklanjanje brojeva
            remove_punctuation: Uklanjanje interpunkcije
            remove_stopwords: Uklanjanje stop words
            stemming: Primena stemminga
            lemmatization: Primena lematizacije
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        # Inicijalizacija
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    
    def clean_text(self, text):
        """
        Osnovno čišćenje teksta
        
        Args:
            text: Ulazni tekst
            
        Returns:
            Očišćen tekst
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Ukloni URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Ukloni mentions (@user)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Ukloni hashtags (#topic)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Ukloni brojeve
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Ukloni interpunkciju
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Ukloni višestruke razmake
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def tokenize(self, text):
        """
        Tokenizacija teksta
        
        Args:
            text: Ulazni tekst
            
        Returns:
            Lista tokena
        """
        return word_tokenize(text)
    
    
    def remove_stop_words(self, tokens):
        """
        Uklanjanje stop words
        
        Args:
            tokens: Lista tokena
            
        Returns:
            Lista tokena bez stop words
        """
        return [token for token in tokens if token not in self.stop_words]
    
    
    def apply_stemming(self, tokens):
        """
        Primena stemminga
        
        Args:
            tokens: Lista tokena
            
        Returns:
            Lista stemovanih tokena
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    
    def apply_lemmatization(self, tokens):
        """
        Primena lematizacije
        
        Args:
            tokens: Lista tokena
            
        Returns:
            Lista lematizovanih tokena
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    
    def preprocess(self, text):
        """
        Kompletan preprocessing pipeline
        
        Args:
            text: Ulazni tekst
            
        Returns:
            Preprocesiran tekst (string)
        """
        # Čišćenje
        text = self.clean_text(text)
        
        # Tokenizacija
        tokens = self.tokenize(text)
        
        # Uklanjanje stop words
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Stemming
        if self.stemming:
            tokens = self.apply_stemming(tokens)
        
        # Lematizacija
        if self.lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        # Spoji tokene nazad u tekst
        return ' '.join(tokens)
    
    
    def preprocess_dataframe(self, df, text_column='text', new_column='cleaned_text'):
        """
        Preprocessing cele kolone u DataFrame-u
        
        Args:
            df: Pandas DataFrame
            text_column: Naziv kolone sa tekstom
            new_column: Naziv nove kolone za očišćen tekst
            
        Returns:
            DataFrame sa novom kolonom
        """
        print(f"Preprocesiranje kolone '{text_column}'...")
        df[new_column] = df[text_column].apply(self.preprocess)
        print("Gotovo!")
        return df


# Primer korišćenja
if __name__ == "__main__":
    # Download resursa
    download_nltk_resources()
    
    # Test
    preprocessor = TextPreprocessor()
    
    test_text = "Bitcoin is going TO THE MOON! 🚀 @elonmusk #BTC #crypto https://example.com"
    
    print("\n" + "="*60)
    print("TEST PREPROCESSINGA")
    print("="*60)
    print("\nOriginalni tekst:")
    print(test_text)
    print("\nPreprocesiran tekst:")
    print(preprocessor.preprocess(test_text))
    print("="*60)