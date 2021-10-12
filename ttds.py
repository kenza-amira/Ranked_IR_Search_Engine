import re
import os
from natsort import natsorted
from nltk.stem import PorterStemmer
import numpy as np
from scipy import special

class SearchEngine(object):
    
    def __init__(self):
        self.stop_words = open("englishST.txt", "r")
        self.ps = PorterStemmer()
        # Initialize the dictionary.
        self.pos_index = {}
 
        # Initialize the file mapping (fileno -> file name).
        self.file_map = {}
        
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
        f = open("input_files/"+ documentName, "r", encoding = "ascii", errors ="surrogateescape")
        
        # Getting output file ready
        out = open("output_files/"+ documentName.replace(".txt","_out.txt"), "w")

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
        
    def inverted_index(self):
        fileno = 0
        file_names = natsorted(os.listdir("output_files"))
        for file in file_names:
            with open("output_files/" + file, 'r', encoding ="ascii", errors ="surrogateescape") as f:
                file_array = f.read().split(' ')
            f.close()
            for pos, term in enumerate(file_array):
                    if term in self.pos_index:
                        # Increment total freq by 1.
                        self.pos_index[term][0] = self.pos_index[term][0] + 1
                        # Check if the term has existed in that DocID before.
                        if fileno in self.pos_index[term][1]:
                            self.pos_index[term][1][fileno].append(pos)
                        else:
                            self.pos_index[term][1][fileno] = [pos]
                    # If term does not exist in the positional index dictionary
                    # (first encounter).
                    else:
                         
                        # Initialize the list.
                        self.pos_index[term] = []
                        # The total frequency is 1.
                        self.pos_index[term].append(1)
                        # The postings list is initially empty.
                        self.pos_index[term].append({})     
                        # Add doc ID to postings list.
                        self.pos_index[term][1][fileno] = [pos]
 
            # Map the file no. to the file name.
            self.file_map[fileno] = "input/" + file
 
            # Increment the file no. counter for document ID mapping             
            fileno += 1


if __name__ == '__main__':
    se = SearchEngine()
    print("preprocessing bible text")
    se.preprocessing("bible.txt")
    print("preprocessing abstracts wiki text")
    se.preprocessing("abstracts.wiki.txt")
    print("Generating inverted index")
    se.inverted_index()
    print("Success! Inverted Positional Index Generated!")
    
    print("For term grass: ")
    print(se.pos_index["grass"])
    
    file_list = se.pos_index["grass"][1]
    print("Filename, [Positions]")
    for fileno, positions in file_list.items():
        print(se.file_map[fileno], positions)