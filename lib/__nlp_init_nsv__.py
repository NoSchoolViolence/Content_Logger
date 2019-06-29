'''
This is the initiation page for NLP. It contains the necessary libraries and some of the sub-routines necessary for NLP.
'''

print('\n', 'Welcome to our Natural Language Processing Library. I am loading at the moment and will be done in a nanosecond...','\n')


# import spacy
# print('Imported spacy')
# import pandas as pd
# print('Imported pandas as pd')
# import numpy as np
# print('Imported numpy as np')
# import nltk
# print('Imported nltk')
# from nltk.tokenize.toktok import ToktokTokenizer
# print('Imported ToktokTokenizer')
# from nltk.stem import PorterStemmer
# print('Imported PortStemmer')
# from nltk.tokenize import sent_tokenize, word_tokenize
# print('Imported sentence and word tokenizer')
# import re
# print('Imported re')
# from bs4 import BeautifulSoup
# print('Imported BeautifulSoup')
# from contractions import CONTRACTION_MAP
# print('Imported CONTRACTION_MAP')
# import unicodedata
# print('Imported unicodedata')
# import PyPDF2
# print('Imported PyPDF2')
# from gensim.summarization import summarize
# print("Imported gensim's summarize")
# import textract
# print('Imported textract')
# from gensim.summarization import keywords
# print("Imported gensim's keywords")
# import pyphen
# print('Imported pyphen')
# from textblob import TextBlob
# print('Imported Textblob')

print('*' *80, '\n')

i = 0

try:
    # Import NLTK if it is installed
    import nltk
    print('nltk imported')
    # This imports NLTK's implementation of the Snowball
    # stemmer algorithm
    from nltk.stem.snowball import SnowballStemmer
    print('nltk SnowballStemmer imported')
    # NLTK's interface to the WordNet lemmatizer
    from nltk.stem.wordnet import WordNetLemmatizer
    print('nltk WordNetLemmatizer imported')
except ImportError:
    nltk = None
    print("NLTK is not installed.")
    i = 1

try:
    # Import spaCy if it is installed
    import spacy
    print("spacy imported")
except ImportError:
    spacy = None
    print("spaCy is not installed.")
    i = 1

try:
    # Import Pattern if it is installed
    from pattern.en import parse
    print('Pattern parse imported')
    import pattern
    print("Pattern imported")
except ImportError:
    parse = None
    print("Pattern is not installed.")
    i = 1

try:
    # Import pandas if it is installed
    import pandas as pd
    print("Pandas imported as pd")
except ImportError:
    pandas = None
    print("Pandas is not installed.")
    i = 1

try:
    # Import numpy if it is installed
    import numpy as np
    print('Numpy imported as np')
except ImportError:
    numpy = None
    print("Numpy is not installed.")
    i = 1    

try:
    # Import re if it is installed
    import re
    print('re imported')
except ImportError:
    re = None
    print("re is not installed.")
    i = 1

try:
    # Import BeautifulSoup if it is installed
    from bs4 import BeautifulSoup
    print('BeautifulSoup imported')
except ImportError:
    BeautifulSoup = None
    print("BeautifulSoup is not installed.")
    i = 1    
    
try:
    # Import CONTRACTION_MAP if it is installed
    from contractions import CONTRACTION_MAP
    print('CONTRACTION_MAP imported')
except ImportError:
    CONTRACTION_MAP = None
    print("CONTRACTION_MAP is not installed.")
    i = 1    
    
try:
    # Import unicodedata if it is installed
    import unicodedata
    print('Unicodedata imported.')
except ImportError:
    unicodedata = None
    print("Unicodedata is not installed.")
    i = 1    

try:
    # Import PyPDF2 if it is installed
    import PyPDF2
    print('PyPDF2 imported.')
except ImportError:
    PyPDF2 = None
    print("PyPDF2 is not installed.")
    i = 1
    
try:
    # Import gensim's summarize if it is installed
    from gensim.summarization import summarize
    print("Gensim's summarize imported.") 
except ImportError:
    summarize = None
    print("Gensim's summarize is not installed.")
    i = 1    
    
try:
    # Import textract if it is installed
    import textract
    print('Textract imported.')
except ImportError:
    textract = None
    print("Textract is not installed.")
    i = 1    

try:
    # Import gensim's keywords if it is installed
    from gensim.summarization import keywords
    print("Gensim's keywords imported")
except ImportError:
    keywords = None
    print("Gensim's keywords is not installed.")
    i = 1    

try:
    # Import pyphen if it is installed
    import pyphen
    print('Pyphen imported.')
except ImportError:
    pyphen = None
    print("PyPhen is not installed.")
    i = 1    

try:
    # Import TextBlob if it is installed
    from textblob import TextBlob
    print('Textblob imported.')
except ImportError:
    TextBlob = None
    print("Textblob is not installed.")
    i = 1    

# Needed for summarizing and reading in text from PDF
try:
    import logging
    print("Logging imported logging for summarizing and reading text from PDFs")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
except ImportError:
    logging = None
    print("Logging is not installed.")
    i = 1    
    
# gives us access to command-line arguments
try:
    # Import sys if it is installed
    import sys
    print("sys imported.")
except ImportError:
    sys = None
    print("sys is not installed.")
    i = 1    

# The Counter collection is a convenient layer on top of
# python's standard dictionary type for counting iterables.
try:
    from collections import Counter
    print("Counter imported.")
except ImportError:
    Counter = None
    print("Counter is not installed.")
    i = 1    

# Import sent_tokenize if it is installed
try:
    from nltk.tokenize import sent_tokenize
    print("NLTK's sent_tokenize imported.")
except ImportError:
    sent_tokenize = None 
    print("sent_tokenizer is not installed.")
    i = 1    

    
# Import word_tokenize if it is installed
try:
    from nltk.tokenize import word_tokenize
    print("NLTK's word_tokenize imported.")
except ImportError:
    word_tokenize = None 
    print("word_tokenizer is not installed.")
    i = 1    

if i < 1:
    print('\n', '_'*50, "\n\n\tOur libraries are loaded correctly."'\n', '_'*50)

print('DONE! Phew, I am good! Record Time!') 
print('\n\nI also loaded the following functions:\n\n\tpull_all_text(df)\n\tadd_to_library(df)\n\tclean_the_text(raw_text)\n\tdrop_nonenglish_words(text)\n\tremove_puntuation(text)\n\tstrip_html_tags(text)\n\tremove_accented_characters(text)\n\texpand_contractions(text, contraction_mapping=CONTRACTION_MAP\n\tremove_special_characters(text, remove_digits=False)\n\tsimple_stemmer(text)\n\tlemmatize_text(text)\n\tremove_stopwords(text, is_lower_case=False)\n\tnormalize_corpus(corpus, html_stripping=True, contraction_expansion=True, accented_char_removal=True, text\n\t\t_lower_case=True, text_lemmatization=True, special_char_removal=True, stopword_removal=True, \n\t\tremove_digits=True)\n\timport_pdf(file_path)\n\ttokenize_by_sentences(text)\n\ttokenize_by_words(text)\n\tfind_keywords(text)\n\tpos_tag(text)\n\tnormalize_corpus(text)\n\timport_pdf(file_path)\n\tword_count(string)\n\tcountwords(words, length_of_word_to_find)\n\tavgwordspersentence(words)\n\tnoofsyllables(words)')


print('\n', "*" * 80, '\n')
print("To learn more about each of these functions, please type 'Tell_me_more(the function's name' and I will happily tell you more about each of these functions.")
print('\n', "*" * 80, '\n')


stopword_list = nltk.corpus.stopwords.words('english')


def pull_all_text(urllibrary):
    import requests
    import pandas as pd
    import time
    
    urllibrary['raw_text'] = 0
    nofills = []
    
    #The TRY wrapper is to suppress EXCEPTIONS that may crop up to alert rather than stop the code. 
    try: 
        index = 0
        for url in urllibrary['url']: #looks at each URL in the urllibrary file
            page = requests.get(url, allow_redirects=False) #pulls the html associated with the URL 
            urllibrary.loc[index,'raw_text'] = page.content #places the raw html into the appropriate row in the dataframe
            urllibrary.loc[index, 'pull_date'] = pd.datetime.today().strftime("%m/%d/%Y")#Time stamps the pull
            print("Processing line", index+1,"of", len(urllibrary['url']), ": text pulled from the", urllibrary.loc[index,'org'],"URL") #This line just lets us know its working

        # This time.sleep() function as a way of putting in a pause so we aren't shut down by ISPs for DOS Attack
            if index%25==0:
                print('\n', '*' *25, '\n',"Pause regulator initiated", '\n', '*' *25)
                time.sleep(3)
                
            index += 1
            
    except Exception as ex:
        print("*" *10, '\n''WARNING: An Exception was thrown at dataset line', index+1,'\n',"*" *10, '\n')
        print(ex)
        print(url, 'deleted.')
        del url
        pass

    
        
'''
****************************************************
The add_to_library(urllibrary) function:

This function looks over our urllibrary for [text] columns with NAN values and pulls the text for those URLs.
****************************************************
'''

def add_to_library(urllibrary):
    import requests
    import pandas as pd
    
    index = 0
    for text in urllibrary['raw_text']: #looks at each URL in the urllibrary file
        if text == 0:
            print('index:', index, 'is NaN')
            page = requests.get(urllibrary[index, 'url']) #pulls the html associated with the URL 
            urllibrary.loc[index,'raw_text'] = page.content #places the raw html into the appropriate row in the dataframe
            urllibrary.loc[index, 'pull_date'] = pd.datetime.today().strftime("%m/%d/%Y")#Time stamps the pull
            print("line", index+1,"of", len(urllibrary['url']), "processed")#This line just lets us know its working
            index +=1
        else:
            print("line", index+1,"of", len(urllibrary['url']), "processed")#This line just lets us know its working
            index +=1
            pass
   
        
# Removing HTML Tags
def strip_html_tags(text):
    import bs4 as BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# Removing Accent characters
def remove_accented_characters(text):
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Expanding Contractions
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
      
      
def clean_the_text(text, remove_numbers=False):
    print('\n', 'CLEANING THE TEXT')
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'lxml')
    
#     print('PRETTYING UP THE TEXT IN THE CLEANING:  ', '\n\t', soup.prettify())
#     text = soup.text
    
    from pattern.web import URL, plaintext
    text = plaintext(text, keep=[], linebreaks=2, indentation=False)

    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    import re
    clean = re.compile(r'^<.*?>}{')
    text = re.sub(clean, '', text)

    text = text.replace('\\xe2\\x80\\x99', "'")
    text = text.replace('\\xc3\\xa9', 'e')
    text = text.replace('\\xe2\\x80\\x90', '-')
    text = text.replace('\\xe2\\x80\\x91', '-')
    text = text.replace('\\xe2\\x80\\x92', '-')
    text = text.replace('\\xe2\\x80\\x93', '-')
    text = text.replace('\\xe2\\x80\\x94', '-')
    text = text.replace('\\xe2\\x80\\x94', '-')
    text = text.replace('\\xe2\\x80\\x98', "'")
    text = text.replace('\\xe2\\x80\\x9b', "'")
    text = text.replace('\\xe2\\x80\\x9c', '"')
    text = text.replace('\\xe2\\x80\\x9c', '"')
    text = text.replace('\\xe2\\x80\\x9d', '"')
    text = text.replace('\\xe2\\x80\\x9e', '"')
    text = text.replace('\\xe2\\x80\\x9f', '"')
    text = text.replace('\\xe2\\x80\\xa6', '...')
    text = text.replace('\\xe2\\x80\\xb2', "'")
    text = text.replace('\\xe2\\x80\\xb3', "'")
    text = text.replace('\\xe2\\x80\\xb4', "'")
    text = text.replace('\\xe2\\x80\\xb5', "'")
    text = text.replace('\\xe2\\x80\\xb6', "'")
    text = text.replace('\\xe2\\x80\\xb7', "'")
    text = text.replace('\\xe2\\x81\\xba', "+")
    text = text.replace('\\xe2\\x81\\xbb', "-")
    text = text.replace('\\xe2\\x81\\xbc', "=")
    text = text.replace('\\xe2\\x81\\xbd', "(")
    text = text.replace('\\xe2\\x81\\xbe', ")")  
    text = text.replace("\'", "'")
    text = text.replace('\\n', ' ')
    text = text.replace('\\xc2\\xae', '')
    text = text.replace('\n',' ')
    text = text.replace('\t','')
    text = text.replace('\s+', '')
    text = text.replace('\r\r\r', '')
    text = text.replace('\\xc2\\xa9 ', '')
    text = text.replace('xe2x80x93', ',')
    text = text.replace('xe2x88x92', '')
    text = text.replace('\\x0c', '')
    text = text.replace('\\xe2\\x80\\x9331', '')
    text = text.replace('xe2x80x94', '')
    text = text.replace('\x0c', ' ')
    text = text.replace(']', '] ')
    text = text.replace('\\xe2\\x80\\x99', "'")
    text = text.replace('xe2x80x99', "'")
    text = text.replace('\\xe2\\x80\\x933', '-')
    text = text.replace('\\xe2\\x80\\x935', '-')
    text = text.replace('\\xef\\x82\\xb7', '')
    text = text.replace('\\', '')
    text = text.replace('xe2x80x99', "'")
    text = text.replace('xe2x80x9cwexe2x80x9d', '')
    text = text.replace('xe2x80x93', ', ')
    text = text.replace('xe2x80x9cEUxe2x80x9d', '')
    text = text.replace('xe2x80x9cxe2x80x9d', '')
    text = text.replace('xe2x80x9cAvastxe2x80x9d', '')
    text = text.replace('xc2xa0', '')
    text = text.replace('xe2x80x9cxe2x80x9d', '')
    text = text.replace('xe2x80x9c', '')
    text = text.replace('xe2x80x9d', '')
    text = text.replace('xc2xad','')
    text = text.replace('x07', '')
    text = text.replace('tttttt', ' ')
    text = text.replace('activetttt.', '')    
    text = text.replace('.sdeUptttt..sdeTogglettttreturn', '') 
    text = text.replace('ttif', '')
    text = text.replace('.ttt.', ' ')
    text = text.replace(" t t ", ' ')
    text = text.replace('tttt ', '')
    text = text.replace(' tt ', ' ')
    text = text.replace(' t ', ' ')
    text = text.replace(' t tt t', ' ')
    text = text.replace('ttt', '')
    text = text.replace('ttr', '')
    text = text.replace('.display', '')
    text = text.replace('div class', '')
    text = text.replace('div id', ' ')
    text = text.replace('Pocy', 'Policy')
    text = text.replace('xc2xa0a', ' ')
    text = text.replace(' b ', '')
    text = text.replace('rrrr', '')
    text = text.replace('rtttr', '')
    text = text.replace('    ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace(' r ', ' ')
    text = text.replace(' tr ', ' ')
    text = text.replace(' rr  r  ', ' ')
    text = text.replace('   tt t t rt ', ' ')
    text = text.replace('r rrr r trr ', ' ')
    text = text.replace(' xe2x80x93 ', ' ')
    text = text.replace(' xe6xa8x82xe9xbdxa1xe6x9cx83  ', ' ')
    text = text.replace(' rrr ', ' ')
    text = text.replace(' rr ', ' ')
    text = text.replace('tr ', '')
    text = text.replace(' r ', '')
    text = text.replace("\'", "")
    text = text.replace(' t* ', ', ')
    text = text.replace('\s+', '')
    text = text.replace('[pic]', '')
    text = text.replace('    ', '')
    text = text.replace('|', '')
    text = text.replace('__', '')
    text = text.replace('b"', '')
    text = text.replace('xe2x80xa2', '. ')
    
    return text

    print('*' *10, 'DROPPING NON-ENGLISH WORDS FROM THE TEXT', '*' *10)
    from nltk.tokenize import word_tokenize
    token_text_w = word_tokenize(text)
    token_text_s = sentence_tokenize(text)
    
    for sentence in token_text_s:
        if ' * ' in sentence:
            token_text_s.strip(sentence)
            
    
    print("*" *10, 'CHECKING SPELLING OF WORDS', "*" *10)
    import enchant
    d = enchant.Dict('en_US')
    bad_words = []

    for word in token_text_w:
        if d.check(word) is not True:
            bad_words.append(word)
            
    bad_words = set(bad_words)
    
    for word in token_text_w:
        if word in bad_words:
            text = text.replace(word, '')
            
    #Trial of a new way of cleaning the text
    index = 0
    print('\n\n', '*' *10, len(tokenize_by_sentences(a)), '*' *10,'\n\n')
    for sent in tokenize_by_sentences(a):
        if 'js' in sent or 'css' in sent or 'png' in sent or'woff2' in sent or ' div ' in sent or ' meta "" ' in sent or 'span' in sent:
            a = a.replace(sent, '')
            print('\n', '*' * 25,'\n','CLEANING TOKENIZED SENTENCES OF CODE IN INDEX', index, '*' * 25)
            index += 1

            
    return (text)  
 
#Dropping words and characters not found in the English Dictionary
def drop_nonenglish_words(text):
    print('*' *10, 'DROPPING NON-ENGLISH WORDS FROM THE TEXT', '*' *10)
    from nltk.tokenize import word_tokenize
    token_text_w = word_tokenize(text)
    
    import enchant
    d = enchant.Dict('en_US')
    bad_words = []

    for word in token_text_w:
        if d.check(word) is not True:
            bad_words.append(word)
            
    bad_words = set(bad_words)
    
    for word in token_text_w:
        if word in bad_words:
            text = text.replace(word, '')
            
    return (text)  

# PUNCTUATION REMOVAL
def remove_punctuation_2(text):
    import string
    table = str.maketrans({key: None for key in string.punctuation})
    return text.translate(table)

def remove_punctuation(text):
    text = re.sub(r'(?u)[^\w\s]', '', text)
    return text


# Stemming
def simple_stemmer(text):
    from nltk.stem import PorterStemmer
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
      
# Lemmatization
def lemmatize_text(text):
    from nltk import nlp
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
      
# Remove Stopwords
def remove_stopwords(text, is_lower_case=False):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenize_by_words(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
      
# Tokenize by words
def tokenize_by_words(text):
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_w = str(word_tokenize(text))
    return str(token_text_w)

#Tokenize by sentences
def tokenize_by_sentences(text):
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_s = str(sent_tokenize(text))
    return str(token_text_s)

#Find Keywords
def find_keywords(text):
    from gensim.summarization import keywords
    key_words = keywords(text)
    key_words = key_words.replace('_', '')
    key_words = key_words.replace('\n', ' ')
    return key_words

#Get POS tags
def pos_tag(text):
    import nltk
    from nltk import pos_tag
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_w = word_tokenize(text)
    return pos_tag(token_text_w)# Normalize Corpus

# Normalize the Corpus
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
      
# Importing the text from a PDF given the pathway to the PDF      
def import_pdf(file_path):
    import textract
    text = textract.process(file_path)
    return text


# count the frequency of words in a sentence
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

#counts number of words of a particular length (dict = string or frequency distribution
def countwords(dic,length):
    tsum=0
    for i in dic:
        if len(i)==length:
            tsum=tsum+dic[i]
            #print(i,dic[i])
    return tsum

# count average number of words per sentence
def avgwordspersentence(words):
    counter=0
    avg=0
    noofsentences=0
    for i in words:
        if(i!='.'):#and i!=','
            counter=counter+1
        else:
            noofsentences+=1
            avg+=counter            
            counter=0
    avg=avg/noofsentences
    return avg



# count the number of words per sentence

def noofsyllabes(corpus):
    import pyphen
    dic = pyphen.Pyphen(lang='en')
    num=0
    for x in corpus:
        s=dic.inserted(x)
        num=num+s.count('-')+1
    return num


def normalize_tokenize(string):
    """
    Takes a string, normalizes it (makes it lowercase and
    removes punctuation), and then splits it into a list of
    words.

    Note that everything in this function is plain Python
    without using NLTK (although as noted below, NLTK provides
    some more sophisticated tokenizers we could have used).
    """
    # make lowercase
    norm = string.lower()

    # remove punctuation
    norm = re.sub(r'(?u)[^\w\s]', '', norm) 

    # split into words
    tokens = norm.split()

    return tokens

def word_form_hapaxes(tokens):
    """
    Takes a list of tokens and returns a list of the
    wordform hapaxes (those wordforms that only appear once)

    For wordforms this is simple enough to do in plain
    Python without an NLP package, especially using the Counter
    type from the collections module (part of the Python
    standard library).
    """

    counts = Counter(tokens) 
    hapaxes = [word for word in counts if counts[word] == 1] 

    return hapaxes


def Tell_me_more(function):
    print("*" * 80)
    if function == pull_all_text:
        print('"pull_all_text(df)" requires a DataFrame with a column named "url". This function looks for that column in your named DataFrame, retrieves the text from that URL, and places it back into the same DataFrame under a column called, "raw_text".', '\n')
    
    elif function == add_to_library:
        print('"add_to_library(df)" looks for new URL entry in the .csv file that has not had its associated HTML pulled, and it pulls the text. It is a lot like "pull_all_text" above, but it does this for new URLs only.', '\n')
    
    elif function == clean_the_text:
        print('"clean_the_text(raw_text)" cleans the raw text of either HTML or PDF. Simply place into the () where the raw_text is located in the DataFrame and it will return a clean version of the text in a new DataFrame column titled, "clean_text".', '\n')
        
    elif function == drop_nonenglish_words:
        print('"drop_nonenglish_words(text) drops all words in the corpus that are not English words.', '\n')
    
    elif function == remove_punctuation:
        print('"remove_puntuation(text)" removes punctuation from the text.', '\n')
    
    elif function == strip_html_tags:
        print('"strip_html_tags(text)" strips the HTML tags from the text.', '\n')
    
    elif function == remove_accented_characters:
        print('"remove_accented_characters(text)" removes characters that have accent marks.', '\n')
    
    elif function == expand_contractions:
        print("'expand_contractions(text, contraction_mapping=CONTRACTION_MAP' expands contractions such as it's to it is.", '\n')
        
    elif function == simple_stemmer:
        print('"simple_stemmer(text)" stems the text simply without having to write out the full code.', '\n')
    
    elif function == lemmatize_text:
        print('"lemmatize_text(text)" lemmatizes the text simply without having to writye out the full code.', '\n')
    
    elif function == remove_stopwords:
        print('"remove_stopwords(text, is_lower_case=False)" removes standard stopwords. There has been no modification of this stopword list.', '\n')
        
    elif function == normalize_corpus:    
        print('"normalize_corpus(corpus, html_stripping=True, contraction_expansion=True, accented_char_removal=True, text_lower_case=True, text_lemmatization=True, special_char_removal=True, stopword_removal=True, remove_digits=True)" normalized the raw text corpus for statistical processing.', '\n')
    
    elif function == import_pdf:
        print('"import_pdf(file_path)" imports the text of a PDF for processing.', '\n')
    
    elif function == tokenize_by_sentences:
        print('"tokenize_by_sentences(text)" tokenizes the corpus by sentences.', '\n')
    
    elif function == tokenize_by_words:
        print('"tokenize_by_words(text)" tokenizes the corpus by words.', '\n')
    
    elif function == find_keywords:
        print('"find_keywords(text)" finds the keywords in a clean corpus and returns them as a list.', '\n')
    
    elif function == pos_tag:
        print('"pos_tag(text)" is our Part-of-Sentence function which returns the part of the sentence each word fulfills in the clean corpus. The Part-of-Sentence output is a list.', '\n')
    
    elif function == normalize_corpus:
        print('"normalize_corpus(text)" normalizes a clean text corpus for statistical processing.', '\n')
    
    elif function == import_pdf:
        print('"import_pdf(file_path)" imports the text of a PDF from a directory.', '\n')
    
    elif function == word_count:
        print('"word_count(string)" takes a string and returns the count of unique words used and how many times each word was used in a corpus.', '\n')
    
    elif function == countwords:
        print('"countwords(text, length_of_word_to_find)" will return the words of a certain length - which you specify - in a corpus. For example, if you want to find all three-letter words, you can now find them easily.', '\n')
    
    elif function == avgwordspersentence:
        print('"avgwordspersentence(text)" returns the average number of words in a sentence within a corpus.', '\n')
    
    elif function == noofsyllables:
        print('"noofsyllables(text)" returns the number of syllables of each word within a corpus.' ,'\n')
    print("*" * 80)


    
def hapaxes_tokens(rawtext):
    #insure the rawtext is under the right variable
    text = rawtext
    
    #tokenize the text
    text = tokenize_by_words(text)
    
    #Remove Punctuation
    text = remove_punctuation(text)
    
    #Normalize and Tokenize text from the sub-routines above
    tokens = normalize_tokenize(text)
    hapaxes = word_form_hapaxes(tokens)  
    
    return hapaxes, tokens
    
    
def nltk_stem_hapaxes(tokens):
    """
    Takes a list of tokens and returns a list of the word
    stem hapaxes.
    """
    if not nltk: 
        # Only run if NLTK is loaded
        return None

    # Apply NLTK's Snowball stemmer algorithm to tokens:
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(token) for token in tokens]

    # Filter down to hapaxes:
    counts = nltk.FreqDist(stems) 
    hapaxes = counts.hapaxes() 
    return hapaxes

    
    
def pt_to_wn(pos):
    """
    Takes a Penn Treebank tag and converts it to an
    appropriate WordNet equivalent for lemmatization.

    A list of Penn Treebank tags is available at:
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    """

    from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

    pos = pos.lower()

    if pos.startswith('jj'):
        tag = ADJ
    elif pos == 'md':
        # Modal auxiliary verbs
        tag = VERB
    elif pos.startswith('rb'):
        tag = ADV
    elif pos.startswith('vb'):
        tag = VERB
    elif pos == 'wrb':
        # Wh-adverb (how, however, whence, whenever...)
        tag = ADV
    else:
        # default to NOUN
        # This is not strictly correct, but it is good
        # enough for lemmatization.
        tag = NOUN

    return tag

def nltk_lemma_hapaxes(tokens):
    """
    Takes a list of tokens and returns a list of the lemma
    hapaxes.
    """
    if not nltk:
        # Only run if NLTK is loaded
        return None

    # Tag tokens with part-of-speech:
    tagged = nltk.pos_tag(tokens) 

    # Convert our Treebank-style tags to WordNet-style tags.
    tagged = [(word, pt_to_wn(tag))
                     for (word, tag) in tagged] 

    # Lemmatize:
    lemmer = WordNetLemmatizer()
    lemmas = [lemmer.lemmatize(token, pos)
                     for (token, pos) in tagged] 

    return nltk_stem_hapaxes(lemmas) 

    
def Freq_Dict(text):
    tokens = normalize_tokenize(text)
    
    wubDict = {}
    for word in tokens:
        if word in wubDict:
                wubDict[word] = wubDict[word] + 1
        else:
                wubDict[word] = 1
            
    return wubDict
    
    
def Freq_Dist(text):
    tokens = normalize_tokenize(text)
    
    wubDict = {}
    for word in tokens:
        if word in wubDict:
                wubDict[word] = wubDict[word] + 1
        else:
                wubDict[word] = 1
            
    len(wubDict) == len(set(tokens))
    if sum(wubDict.values()) == len(tokens):
        print("Our Dictionary and our Tokens match!")
    else:
        print('We have an error in the length of the dictionary and the tokens.')

    from nltk.probability import FreqDist
    wubFD = FreqDist(word for word in tokens)
    print(wubFD.items())
    
    print('\nOur top 10 most frequent words are:\n\t')
    print(wubFD.tabulate(10))
    
    import matplotlib
    wubFD.plot(20)
    
    
def wordcloud(text):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    wubCloud = WordCloud().generate(text)
    plt.figure()
    plt.imshow(wubCloud)
    plt.axis("off")
    
# Hapaxes using spaCy
def spacy_hapaxes(rawtext):
    """
    Takes plain text and returns a list of lemma hapaxes using
    the spaCy NLP package.
    """
    if not spacy:
        # Only run if spaCy is installed
        return None

    # Load the English spaCy parser
    spacy_parse = spacy.load('en')

    # Tokenize, parse, and tag text:
    doc = spacy_parse(rawtext)

    lemmas = [token.lemma_ for token in doc
            if not token.is_punct and not token.is_space] 

    # Now we can get a count of every lemma:
    counts = Counter(lemmas) 

    # We are interested in lemmas which appear only once
    hapaxes = [lemma for lemma in counts if counts[lemma] == 1]
    return hapaxes

#Hapaxes using Pattern
def pattern_hapaxes(rawtext):
    """
    Takes plain text and returns a list of lemma hapaxes
    using the Pattern NLP module.
    """
    if not parse:
        # Only run if Pattern is installed
        return None

    sentences = parse(rawtext, lemmata=True,
                     tags=False, chunks=False) 
    sentences = sentences.split() 

    # Iterate through each word of each sentence and collect
    # the lemmas (which is the last item of each word):
    lemmas = []
    for sentence in sentences:
        for word in sentence:
            lemmas.append(word[-1])

    # Count the lemmas
    counts = Counter(lemmas)

    # Find hapaxes
    hapaxes = [lemma for lemma in counts if counts[lemma] == 1]
    return hapaxes


