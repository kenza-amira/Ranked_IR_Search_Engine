import re
import os
from natsort import natsorted
from nltk.stem import PorterStemmer
import pprint
import math
import numpy as np


class SearchEngine(object):

    def __init__(self):
        self.stop_words = open("englishST.txt", "r")
        self.ps = PorterStemmer()
        self.pp = pprint.PrettyPrinter(indent=2)
        # Initialize the dictionary.
        self.pos_index = {}
        # Initialize the file mapping (fileno -> file name).
        self.file_map = {}

        # Boolean results initialization
        self.boolRes = []

        # Ranked results initialization
        self.rankedRes = []

        # Getting List of Stop word's ready: Getting rid of punctuation
        # i.e you'll becomes youll
        self.st_words = []
        words = self.stop_words.readlines()
        for word in words:
            word = word.strip()
            word = re.findall('\\w+', word)
            self.st_words.append(''.join(word))

    def preprocessing(self, documentName):
        # Opening the file
        f = open("input_files/" + documentName, "r",
                 encoding="ascii", errors="surrogateescape")

        # Getting output file ready
        out = open("output_files/" +
                   documentName.replace(".txt", "_out.txt"), "w")

        # Reading the file line by line
        lines = f.readlines()
        for line in lines:
            # Tokenization
            res = re.findall('\\w+', line)
            # Stemming + Stopping + Case Folding
            res = ' '.join([self.ps.stem(i.lower())
                           for i in res if i not in self.st_words])
            res += ' '
            # Writing processed output to output file
            out.write(res)
        out.close()

    def inverted_index(self):
        fileno = 0
        file_names = natsorted(os.listdir("output_files"))
        for file in file_names:
            with open("output_files/" + file, 'r', encoding="ascii",
                      errors="surrogateescape") as f:
                file_array = f.read().split(' ')
            f.close()
            for pos, term in enumerate(file_array):
                if term in self.pos_index and term != '':
                    # Check if the term has existed in that DocID before.
                    if fileno in self.pos_index[term][1]:
                        self.pos_index[term][1][fileno].append(pos)
                    else:
                        self.pos_index[term][1][fileno] = [pos]
                        self.pos_index[term][0] = self.pos_index[term][0] + 1
                    # If term does not exist in the positional index dictionary
                    # (first encounter).
                elif term != '':
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

    def writeIndexToFile(self):
        out = open("index.txt", "w")
        for term in (self.pos_index):
            out.write(term + ":" + str(self.pos_index[term][0]) + "\n")
            for file_no in self.pos_index[term][1]:
                out.write("\t" + str(file_no) + ":" +
                          str(self.pos_index[term][1][file_no])
                          .replace('[', '')
                          .replace(']', '') + "\n")
        out.close()

    def phraseSearch(self, query, out=0):
        doc_keeper = []
        terms = query.replace("\"", "").split()
        term1 = terms[0]
        term2 = terms[1]

        # TOKENIZATION + CASE FOLDING + STEMMING
        term1 = ''.join(re.findall('\\w+', term1))
        term1 = self.ps.stem(term1.lower())
        term2 = ''.join(re.findall('\\w+', term2))
        term2 = self.ps.stem(term2.lower())

        # SEARCH IN POSITIONAL INDEX ONLY IF BOTH TERMS EXIST
        if (term1 in self.pos_index and term2 in self.pos_index):
            # GETTING THE DOC IDS THAT CONTAIN BOTH TERM
            out1 = [doc_id for doc_id in self.pos_index[term1][1]]
            out2 = [doc_id for doc_id in self.pos_index[term2][1]]
            intersec = out1 and out2
            # IF THE INTERSECTION IS NULL WE STOP THE SEARCH
            if intersec == []:
                return "No results"
            for i in intersec:
                # GETTING THE TERMS POSITIONS FOR EACH DOC
                positions_1 = self.pos_index[term1][1][i]
                positions_2 = self.pos_index[term2][1][i]

                len1 = len(positions_1)
                len2 = len(positions_2)
                k = j = 0
                while k != len1:
                    while j != len2:
                        if abs(positions_1[k] - positions_2[j] == 1):
                            if (not(query[0], str(i)) in self.boolRes
                                    and int(i)not in doc_keeper):
                                if out == 0:
                                    self.boolRes.append((query[0], str(i)))
                                doc_keeper.append(int(i))
                        elif positions_2[j] > positions_1[k]:
                            break
                        j += 1
                    k += 1
        return doc_keeper

    def proximitySearch(self, query):
        # FINDING INTEGER PROXIMITY
        proximity = re.search('#(.*)\\(', query)
        proximity = int(proximity.group(1))

        # ISOLATING FIRST TERM
        term1 = re.search('\\((.*)\\,', query)
        term1 = term1.group(1)

        # ISOLATING SECOND TERM
        term2 = re.search('\\,(.*)\\)', query)
        term2 = term2.group(1)

        # TOKENIZATION + CASE FOLDING + STEMMING
        term1 = ''.join(re.findall('\\w+', term1))
        term1 = self.ps.stem(term1.lower())
        term2 = ''.join(re.findall('\\w+', term2))
        term2 = self.ps.stem(term2.lower())

        # SEARCH IN POSITIONAL INDEX ONLY IF BOTH TERMS EXIST
        if (term1 in self.pos_index and term2 in self.pos_index):
            # GETTING THE DOC IDS THAT CONTAIN BOTH TERM
            out1 = [doc_id for doc_id in self.pos_index[term1][1]]
            out2 = [doc_id for doc_id in self.pos_index[term2][1]]
            intersec = out1 and out2
            # IF THE INTERSECTION IS NULL WE STOP THE SEARCH
            if intersec == []:
                return "No results"
            for i in intersec:
                # GETTING THE TERMS POSITIONS FOR EACH DOC
                positions_1 = self.pos_index[term1][1][i]
                positions_2 = self.pos_index[term2][1][i]

                len1 = len(positions_1)
                len2 = len(positions_2)
                k = j = 0
                while k != len1:
                    while j != len2:
                        if abs(positions_1[k] - positions_2[j] <= proximity):
                            if not(query[0], str(i)) in self.boolRes:
                                self.boolRes.append((query[0], str(i)))
                        elif positions_2[j] > positions_1[k]:
                            break
                        j += 1
                    k += 1

    def notOperation(self, term, flag=0):
        no_of_files_list = list(range(0, len(os.listdir('input_files'))))
        if flag == 0:
            return [x for x in no_of_files_list
                    if x not in self.pos_index[term][1]]
        else:
            return [x for x in no_of_files_list if x not in term]

    def booleanSearch(self, query):
        if "AND" in query:
            terms = dict()
            new_query = query[2:].strip()
            term1 = re.search('AND (.*)', new_query).group(1).strip()
            term2 = re.search('(.*) AND', new_query).group(1).strip()
            terms[term1] = []
            terms[term2] = []
            checker = term1.replace("NOT ", "").replace("\"", "").strip()
            checker += " "
            checker += term2.replace("NOT ", "").replace("\"", "").strip()
            checker = self.ps.stem(checker.strip().lower()).split(' ')
            if (all(c in self.pos_index for c in checker)):
                for term in terms:
                    if "NOT" in term:
                        not_term = re.search('NOT(.*)', term).group(1).strip()
                        if "\"" in not_term:
                            not_term = self.phraseSearch(not_term, out=1)
                            terms[term] = self.notOperation(not_term, flag=1)
                        else:
                            not_term = self.ps.stem(not_term.lower())
                            terms[term] = self.notOperation(not_term)
                    else:
                        if "\"" in term:
                            terms[term] = self.phraseSearch(term, out=1)
                        else:
                            tmp_term = self.ps.stem(term.lower())
                            terms[term] = [d for d in
                                           self.pos_index[tmp_term][1]]
                and_list = terms[term1] and terms[term2]
                if and_list != []:
                    for item in and_list:
                        self.boolRes.append((query[0], str(item)))
        elif "OR" in query:
            terms = dict()
            new_query = query[2:].strip()
            term1 = re.search('OR (.*)', new_query).group(1).strip()
            term2 = re.search('(.*) OR', new_query).group(1).strip()
            terms[term1] = []
            terms[term2] = []
            checker = term1.replace("NOT ", "").replace("\"", "")
            checker += " "
            checker += term2.replace("NOT ", "").replace("\"", "")
            checker = self.ps.stem(checker.strip().lower()).split(' ')
            if (all(c in self.pos_index for c in checker)):
                for term in terms:
                    if "NOT" in term:
                        not_term = re.search('NOT(.*)', term).group(1).strip()
                        if "\"" in not_term:
                            not_term = self.phraseSearch(not_term, out=1)
                            terms[term] = self.notOperation(not_term, flag=1)
                        else:
                            not_term = self.ps.stem(not_term.lower())
                            terms[term] = self.notOperation(not_term)
                    else:
                        tmp_term = self.ps.stem(term.lower())
                        if "\"" in term:
                            terms[term] = self.phraseSearch(term, out=1)
                        else:
                            terms[term] = [d for d
                                           in self.pos_index[tmp_term][1]]
                or_list = terms[term1] + list(set(terms[term2])
                                              - set(terms[term1]))
                if or_list != []:
                    for item in or_list:
                        self.boolRes.append((query[0], str(item)))

    def booleanQueryFile(self, file="queries.boolean.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            if "AND" in line or "OR" in line or "NOT" in line:
                self.booleanSearch(line)
            elif "#" in line:
                self.proximitySearch(line)
            elif "\"" in line:
                self.phraseSearch(line)
            # CASE WHERE THE QUERY ONLY CONTAINS A SINGLE WORD
            elif len(line[2:].split(' ')) == 1:
                docs = self.pos_index[line[2:].strip()][1]
                for d in docs:
                    self.boolRes.append((line[0], str(d)))

    def writeBooleanToFile(self):
        out = open("results.boolean.txt", "w")
        for a, b in self.boolRes:
            out.write(a + "," + b + "\n")
        out.close()

    def normalize(self, scores):
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def tfidfSearch(self, query):
        terms = [self.ps.stem(t) for t in query[2:].strip().lower().split(' ')]
        N = len(os.listdir('input_files'))
        scores = dict()
        for t in terms:
            df = self.pos_index[t][0]
            idf = math.log10(N/df)
            for doc in list(range(0, N)):
                try:
                    tf = len(self.pos_index[t][1][doc])
                    tmp = 1 + math.log10(tf)
                    try:
                        scores[doc] += tmp*idf
                    except KeyError:
                        scores[doc] = tmp*idf
                except KeyError:
                    tf = 0.00000001
                    tmp = 1 + math.log10(tf)
                    try:
                        scores[doc] += tmp*idf
                    except KeyError:
                        scores[doc] = tmp*idf
        score_vals = self.normalize(list(scores.values()))
        i = 0
        for doc in scores:
            scores[doc] = score_vals[i]
            i += 1
        scores = dict(sorted(scores.items(), key=lambda item: item[1],
                             reverse=True))
        self.rankedRes.append((query[0], list(scores.items())[:150]))

    def rankedQueryFile(self, file="queries.ranked.txt"):
        f = open(file)
        lines = f.readlines()
        out = open("results.ranked.txt", "w")
        for line in lines:
            self.tfidfSearch(line)

        for q, res in self.rankedRes:
            for r in res:
                out.write(q + "," + str(r[0]) + "," + str(r[1]) + "\n")
        out.close()


if __name__ == '__main__':
    se = SearchEngine()
    # print("preprocessing bible text")
    # se.preprocessing("bible.txt")
    # print("preprocessing abstracts wiki text")
    # se.preprocessing("abstracts.wiki.txt")
    print("Generating inverted index")
    se.inverted_index()
    print("Success! Inverted Positional Index Generated!")
    se.writeIndexToFile()
    print("Running Boolean queries")
    se.booleanQueryFile()
    se.writeBooleanToFile()
    print("Success! Results at: results.boolean.txt")
    print("Running Ranked Retrival queries")
    se.rankedQueryFile()
    print("Success! Results at: results.ranked.txt")
