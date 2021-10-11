import re
from nltk.stem import PorterStemmer
import numpy as np
from matplotlib import pyplot as plt
from scipy import special

class SearchEngine(object):
    
    def __init__(self):
        self.stop_words = open("englishST.txt", "r")
        self.ps = PorterStemmer()
        
        # Getting List of Stop word's ready: Getting rid of punctuation
        # i.e you'll becomes youll
        self.st_words = []
        words = self.stop_words.readlines()
        for word in words:
            word = word.strip()
            word = re.findall('\w+',word)
            self.st_words.append(''.join(word))
    
    def preprocessing(self, documentName):
        # Opening the file
        f = open(documentName, "r")
        
        # Getting output file ready
        out = open(documentName.replace(".txt","_out.txt"), "w")

        # Reading the file line by line
        lines = f.readlines()
        for line in lines:
            # Tokenization
            res = re.findall('\w+', line)
            # Stemming + Stopping + Case Folding
            res = ' '.join([self.ps.stem(i.lower()) for i in res if i not in self.st_words])
            res += ' '
            # Writing processed output to output file
            out.write(res)
        out.close()
        
        

        # frequency = {}
        # open_file = open(r"C:\Users\kenza\Downloads\bible_out.txt", 'r')
        # file_to_string = open_file.read()
        # words = re.findall(r'(\b[A-Za-z][a-z]{2,9}\b)', file_to_string)

        # for word in words:
        #     count = frequency.get(word,0)
        #     frequency[word] = count + 1

        # counts = frequency.values()
        # tokens = frequency.keys()

        # s = np.array(counts)
        # from scipy.stats import itemfreq
        # tmp = itemfreq(counts)
if __name__ == '__main__':
    se = SearchEngine()
    se.preprocessing("bible.txt")