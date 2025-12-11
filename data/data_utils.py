import re

def clean_data(line):
    """
    Cleans a single data line by removing duplicate triplets.
    
    Args:
        line (str): An input line containing tokens and triplets, separated by '####'.
        
    Returns:
        str: The cleaned line with duplicate triplets removed.
    """
    # Split the line into tokens and the triplet string
    tokens, triplets_str = line.strip().split('####')
    # Remove duplicate triplets by converting them to a set and back
    unique_triplets = list(set([str(t) for t in eval(triplets_str)]))
    return f"{tokens}####{str([eval(t) for t in unique_triplets])}\n"


def line_to_dict(line, is_clean=False):
    """
    Converts a single data line into a dictionary.
    
    Args:
        line (str): An input line with tokens and triplets.
        is_clean (bool): Whether to clean the data line before processing.
        
    Returns:
        dict: A dictionary with 'token' and 'triplets' keys.
    """
    if is_clean:
        line = clean_data(line)
    
    # Split the line into the sentence and the triplet string
    sentence, triplets_str = line.strip().split('####')
    
    # Convert triplet format to use start-end indices for aspect and opinion terms
    triplets = []
    for t in eval(triplets_str):
        triplets.append(
            ([t[0][0], t[0][-1]], [t[1][0], t[1][-1]], t[2])
        )
    
    # Sort triplets by the aspect's start index and the opinion's end index
    triplets.sort(key=lambda x: (x[0][0], x[1][-1]))
    
    return {'token': sentence.split(' '), 'triplets': triplets}

def normalize_arabic(text):
    """
    Normalizes Arabic text by standardizing character variations.
    
    This function performs several common Arabic text normalizations:
    - Standardizes different forms of alef (أ, إ, آ) to a plain alef (ا).
    - Converts taa marbuta (ة) to haa (ه).
    - Normalizes alef maksura (ى) to yaa (ي).
    - Removes diacritics (harakat).
    
    Args:
        text (str): The Arabic text to normalize.
        
    Returns:
        str: The normalized Arabic text.
    """
    if not text:
        return text
    
    # Normalize alef variations to a single form
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Normalize taa marbuta to haa
    text = re.sub(r'ة', 'ه', text)
    # Normalize alef maksura to yaa
    text = re.sub(r'ى', 'ي', text)
    # Normalize hamza on waw to waw
    text = re.sub(r'ؤ', 'و', text)
    # Normalize hamza on yaa to yaa
    text = re.sub(r'ئ', 'ي', text)
    
    # Remove diacritics (harakat)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    # Convert Arabic numerals to Western numerals
    arabic_to_western = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
                         '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
    for ar, en in arabic_to_western.items():
        text = text.replace(ar, en)
    
    # Remove the tatweel (kashida) character
    text = re.sub(r'ـ', '', text)
    
    # Normalize spaces and remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_arabic_tokens(tokens):
    """
    Applies Arabic normalization to a list of tokens.
    
    Args:
        tokens (list): A list of tokens to be normalized.
        
    Returns:
        list: A list of normalized tokens.
    """
    return [normalize_arabic(token) for token in tokens]
