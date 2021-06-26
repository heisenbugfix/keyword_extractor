# keyword_extractor
Input a raw text file, get top relevant keywords - based on count stat

Usage:
`python keyword_extractor.py <path to text file>`

Requirements:
```
nltk
```
Install punkt, stopwords and averaged_perceptron_tagger for nltk.
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```
