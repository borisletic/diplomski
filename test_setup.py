"""
Test skript - provera da li sve radi
Pokreni: python test_setup.py
"""

import sys

print("\n" + "="*60)
print("PROVERA INSTALACIJE - Diplomski Rad")
print("="*60 + "\n")

# Test 1: Python verzija
print("1️⃣  Python verzija:")
print(f"   {sys.version}")
if sys.version_info < (3, 8):
    print("   ⚠️  UPOZORENJE: Python 3.8+ je preporučen!")
else:
    print("   ✅ OK\n")

# Test 2: Osnovne biblioteke
print("2️⃣  Osnovne biblioteke:")
try:
    import pandas
    print(f"   ✅ pandas {pandas.__version__}")
except ImportError:
    print("   ❌ pandas nije instaliran!")

try:
    import numpy
    print(f"   ✅ numpy {numpy.__version__}")
except ImportError:
    print("   ❌ numpy nije instaliran!")

try:
    import matplotlib
    print(f"   ✅ matplotlib {matplotlib.__version__}")
except ImportError:
    print("   ❌ matplotlib nije instaliran!")

# Test 3: ML biblioteke
print("\n3️⃣  Machine Learning biblioteke:")
try:
    import sklearn
    print(f"   ✅ scikit-learn {sklearn.__version__}")
except ImportError:
    print("   ❌ scikit-learn nije instaliran!")

try:
    import nltk
    print(f"   ✅ nltk {nltk.__version__}")
except ImportError:
    print("   ❌ nltk nije instaliran!")

# Test 4: Deep Learning
print("\n4️⃣  Deep Learning biblioteke:")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   🎮 CUDA dostupan! GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   💻 CUDA nije dostupan (samo CPU)")
except ImportError:
    print("   ❌ PyTorch nije instaliran!")

try:
    import transformers
    print(f"   ✅ transformers {transformers.__version__}")
except ImportError:
    print("   ❌ transformers nije instaliran!")

# Test 5: Dodatno
print("\n5️⃣  Dodatne biblioteke:")
try:
    import yfinance
    print(f"   ✅ yfinance {yfinance.__version__}")
except ImportError:
    print("   ❌ yfinance nije instaliran!")

try:
    from wordcloud import WordCloud
    print(f"   ✅ wordcloud")
except ImportError:
    print("   ❌ wordcloud nije instaliran!")

try:
    import seaborn
    print(f"   ✅ seaborn {seaborn.__version__}")
except ImportError:
    print("   ❌ seaborn nije instaliran!")

# Test 6: Struktura foldera
print("\n6️⃣  Struktura projekta:")
import os

folders = ['data/raw', 'data/processed', 'notebooks', 'src', 'results']
for folder in folders:
    if os.path.exists(folder):
        print(f"   ✅ {folder}/")
    else:
        print(f"   ⚠️  {folder}/ ne postoji!")

# Test 7: NLTK resursi
print("\n7️⃣  NLTK resursi:")
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    print("   ✅ punkt")
except:
    print("   ⚠️  punkt nije preuzet - pokreni: python -c \"import nltk; nltk.download('punkt')\"")

try:
    import nltk
    nltk.data.find('corpora/stopwords')
    print("   ✅ stopwords")
except:
    print("   ⚠️  stopwords nisu preuzeti - pokreni: python -c \"import nltk; nltk.download('stopwords')\"")

try:
    import nltk
    nltk.data.find('corpora/wordnet')
    print("   ✅ wordnet")
except:
    print("   ⚠️  wordnet nije preuzet - pokreni: python -c \"import nltk; nltk.download('wordnet')\"")

# Test 8: Src moduli
print("\n8️⃣  Projekat moduli:")
try:
    sys.path.append('src')
    import preprocessing
    print("   ✅ preprocessing.py")
except ImportError as e:
    print(f"   ❌ preprocessing.py greška: {e}")

try:
    import models
    print("   ✅ models.py")
except ImportError as e:
    print(f"   ❌ models.py greška: {e}")

try:
    import evaluation
    print("   ✅ evaluation.py")
except ImportError as e:
    print(f"   ❌ evaluation.py greška: {e}")

try:
    import utils
    print("   ✅ utils.py")
except ImportError as e:
    print(f"   ❌ utils.py greška: {e}")

# Finalna poruka
print("\n" + "="*60)
print("PROVERA ZAVRŠENA!")
print("="*60)

print("\n💡 Sledeći korak:")
print("   1. Preuzmi Bitcoin Twitter dataset sa Kaggle")
print("   2. Sačuvaj u: data/raw/bitcoin_tweets.csv")
print("   3. Pokreni: jupyter notebook")
print("   4. Otvori: notebooks/01_data_exploration.ipynb")
print("\n✨ Srećno sa diplomskim!\n")