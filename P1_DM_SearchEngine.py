"""
Search engine based on the cosine similarity score on the transcript of inaugural addresses by different US presidents dataset.

Author: Sudharani Bannengala
"""
# import necessary libraries
import os
from nltk.tokenize import RegexpTokenizer
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import math

# Initialize global variables
filepath = './US_Inaugural_Addresses'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
doc_count = 0
docDict = {} # dictionary containing preprocessed document tokens with filename as key value
tf = {} # contains term frequency
    

def Preprocessing(doc):
    """
    Preprocess (lowercase, tokenize and stemming) the text document.
        Parameters:
        -----------
        doc: string
            contains the text content.        
    """
    #convert text content to lowercase
    doc_lower = doc.lower()
    #tokenize the input text content
    doc_token = tokenizer.tokenize(doc_lower)
    #remove stop words from the text content from document and apply stemming
    doc_remstword = [stemmer.stem(word) for word in doc_token if word not in stop_words]
    return doc_remstword



# Function to load the data from the provided filepath
def loadPrepareData(filepath):
    """
    loads and preprocess the data from the provided filepath.
        Parameters:
        -----------
        filepath: string
            contains the dataset path from which file content has to be loaded.        
    """
    #initialize the document dictionary
    global doc_count
    corpusroot = filepath
    #loop to fetch all files starting from '01', '02', '03' and '04'
    for filename in os.listdir(corpusroot):
        if filename.startswith('0') or filename.startswith('1') or filename.startswith('2') or filename.startswith('3'):
            file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
            doc = file.read()
            #calculate term frequency and store in dict w.r.t filename
            word_tokens = Preprocessing(doc)
            tf[filename] = Counter(word_tokens)
            doc_count += 1
            file.close()
            

def get_magnitude(filename):
    """
    loads and preprocess the data from the provided filepath.
        Parameters:
        -----------
        filepath: string
            contains the dataset path from which file content has to be loaded.        
    """
    sum_n = 0
    # loop to calculate total log weighted tf-idf sum for all tokens in a file
    for token in tf[filename]:
        idf_score = getidf(token, True)
        term_freq = tf[filename][token]
        tf_score = 1 + math.log10(term_freq)
        sum_n += (idf_score * tf_score)**2
    magnitude = math.sqrt(sum_n)
    return magnitude
        


def getidf(token, stemmed=False):
    """
    Get the idf score for the given token.
        Parameters:
        -----------
        token: string
            contains single word.
        stemmed: boolean
            represents if the given token is already stemmed or not.
    """
    df = 0
    # apply porter stemming on the token only if its not stemmed already
    if stemmed:
        stemmed_word = token
    else:
        stemmed_word = stemmer.stem(token)
    
    #calculate document frequency of the token
    for term in tf.values():
        if stemmed_word in term:
            df +=1
    if df == 0:
        return -1  # return -1 if word is not present in the corpus
    idf = math.log10(doc_count / df)
    return idf


def getweight(filename, token, stemmed=False):
    """
    Get the tf-idf score of the given token.
        Parameters:
        -----------
        token: string
            contains single word.
        stemmed: boolean
            represents if the given token is already stemmed or not.
    """
    if filename not in tf:
        return 0 # if filename not in corpus, return 0
    # apply porter stemming on the token only if its not stemmed already
    if stemmed:
        stemmed_word = token
    else:
        stemmed_word = stemmer.stem(token)  
    # calculate idf score for stemmed token
    idf_score = getidf(stemmed_word, True)
    term_freq = tf[filename][stemmed_word]
    
    # calculate log weighted term frequency
    if term_freq == 0:
        return 0     # when token is not in a given document, return 0
    else:
        tf_score = 1 + math.log10(term_freq)
        
    # calculate normalized tf-idf score  
    tf_idf_score = tf_score * idf_score
    mag = get_magnitude(filename)
    norm_tf_idf = tf_idf_score / mag
    
    return norm_tf_idf


def query(qstring):
    """
    Get the most similarity score with filename for the given query string.
        Parameters:
        -----------
        qstring: string
            contains one or more words of query.
    """
    qs_weights = {}
    norm_sum = 0
    # Preprocess the given query string
    qs_tokens = Preprocessing(qstring)
    # calculate query term frequency
    query_term_freq = Counter(qs_tokens)

    #loop to calculate log weighted tf value and determinant to perform cosine normalization
    for token, freq in query_term_freq.items():
        query_term_freq[token] = 1 + math.log10(freq)
        norm_sum += (query_term_freq[token]**2)   #df score is considered as 1 for tf*idf calculation
    query_det = math.sqrt(norm_sum)

    #loop to calculate the similarity score for each query token in each file of corpus
    for filename, tf_score in tf.items():
        sim_score = 0
        for token in query_term_freq:
            weight = getweight(filename, token, True)
            sim_score += (weight * (query_term_freq[token]/query_det))
        qs_weights[filename] = sim_score
    #if query token is not present in any document, return 0  
    all_zero = all(value == 0 for file, value in qs_weights.items())
    if all_zero == True:
        return ("query string doesn't exist in any document",0)

    #find max similarity score
    max_doc_score = max(qs_weights.items(), key = lambda val: val[1])

    return max_doc_score


    

def main():
    """
    Main function to initialize load and prepare the data
    """
    # Calculating the tf scores for the corpus data
    loadPrepareData(filepath)

# execute main function
if __name__ == "__main__":
    main()