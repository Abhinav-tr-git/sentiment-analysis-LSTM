import re
import string

def clean_text(text):
    """
    Standardizes and cleans input text for sentiment analysis.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags (e.g., <br />)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
