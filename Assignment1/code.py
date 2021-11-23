import re
import os
from natsort import natsorted
# from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import pprint
import math
import numpy as np
import xml.etree.ElementTree as ET
import shutil


class SearchEngine(object):

    def __init__(self):
        self.stop_words = open("englishST.txt", "r")
        self.ps = SnowballStemmer(language='english')
        self.pp = pprint.PrettyPrinter(indent=2)
        # Initialize the dictionary.
        self.inv_index = {}

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

    def splittingDocs(self):
        """
        This function splits the trec collection into multiple
        input docs named according to the DOCNO in the XML file.
        I used an XML parser from a python library.
        """
        os.mkdir('input_files')
        os.mkdir('output_files')
        p = r'C:\Users\kenza\OneDrive\Documents\TTDS\collections\trec.5000.xml'
        tree = ET.parse(p)
        root = tree.getroot()
        for doc in root.findall("DOC"):
            doc_no = doc.find("DOCNO").text
            headline = doc.find("HEADLINE").text
            text = doc.find("TEXT").text
            f = open("input_files/" + str(doc_no)+".txt", "w")
            f.write(headline + " " + text)
            f.close()

    def preprocessing(self, documentName):
        """
        This function does all the preprocessing work and saves
        the new preprocessed docs in the output files:
        Line by line we tokenise then we do the stemming (using
        a Porter Stemmer), the stopping and the case folding.

        Args:
            documentName (String documentName): This function
            only takes a documentName
        """
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
                           for i in res if i.lower() not in self.st_words])
            res += ' '
            # Writing processed output to output file
            out.write(res)
        out.close()

    def inverted_index(self):
        """
        This function generates the inverted positional index.
        - We iterate through the output files as we only want to
        create an index of the already pre processed words.
        - We then loop through the file checking terms and indices.
        - If the term alrady has existed in a DocID before then we
        just add another position
        - If it has never appeared before, we initialise the position
        array for that term with the index we are at and we increase
        the frequency
        - If we have never encountered the term bedore then we initialize
        the list for that term, we set the frequency to 1 and add the doc
        to the list of documents.
        """
        file_names = natsorted(os.listdir("output_files"))
        for file in file_names:
            with open("output_files/" + file, 'r', encoding="ascii",
                      errors="surrogateescape") as f:
                file_array = f.read().split(' ')
            f.close()
            fileno = int(file.replace("_out.txt", ""))
            for pos, term in enumerate(file_array):
                if term in self.inv_index and term != '':
                    # Check if the term has existed in that DocID before.
                    if fileno in self.inv_index[term][1]:
                        # If it has, append the position
                        self.inv_index[term][1][fileno].append(pos)
                    else:
                        # Otherwise, initialize the positional list
                        # and increase the frequency
                        self.inv_index[term][1][fileno] = [pos]
                        self.inv_index[term][0] = self.inv_index[term][0] + 1
                    # If term does not exist in the positional index dictionary
                    # (first encounter).
                elif term != '':
                    # Initialize the list.
                    self.inv_index[term] = []
                    # Initialize frequency to 1
                    self.inv_index[term].append(1)
                    # Initialize docs lists
                    self.inv_index[term].append({})
                    # Add doc ID and pos to docs list
                    self.inv_index[term][1][fileno] = [pos]

    def writeIndexToFile(self):
        """
        This function writes the index to a file named index.txt
        """
        out = open("index.txt", "w")
        for term in (self.inv_index):
            out.write(term + ":" + str(self.inv_index[term][0]) + "\n")
            for file_no in self.inv_index[term][1]:
                out.write("\t" + str(file_no) + ": " +
                          str(self.inv_index[term][1][file_no])
                          .replace('[', '')
                          .replace(']', '')
                          .replace(' ', '') + "\n")
        out.close()

    def phraseSearch(self, query, out=0):
        """
        This function takes care of phrase search. Where the input looks like
        "middle east". It updates the boolRes (storing the output for boolean
        searches) list with a tuple of the form (query_no, document).

        Args:
            query (String): This function takes the query string as an argument
            out (int, optional): This is a flag that defaults to 0 when the
            function isn't called from within the BooleanSearch function. If it
            is set to 1 then it means we are calling the function from within
            the Boolean Search. The only difference is that we won't update the
            boolRes immediately.

        Returns:
            list: This function returns a list of documents containing the docs
            that match the query. We only use that if out=1.
        """
        doc_keeper = []
        terms = query[1:].replace("\"", "").split()
        term1 = terms[0]
        term2 = terms[1]

        # TOKENIZATION + CASE FOLDING + STEMMING
        term1 = ''.join(re.findall('\\w+', term1))
        term1 = self.ps.stem(term1.lower())
        term2 = ''.join(re.findall('\\w+', term2))
        term2 = self.ps.stem(term2.lower())

        # SEARCH IN POSITIONAL INDEX ONLY IF BOTH TERMS EXIST
        if (term1 in self.inv_index and term2 in self.inv_index):
            # GETTING THE DOC IDS THAT CONTAIN BOTH TERM
            out1 = [doc_id for doc_id in self.inv_index[term1][1]]
            out2 = [doc_id for doc_id in self.inv_index[term2][1]]
            intersec = list(set(out1) & set(out2))
            # IF THE INTERSECTION IS NULL WE STOP THE SEARCH
            if intersec == []:
                return []
            for i in intersec:
                # GETTING THE TERMS POSITIONS FOR EACH DOC
                positions_1 = self.inv_index[term1][1][i]
                positions_2 = self.inv_index[term2][1][i]

                len1 = len(positions_1)
                len2 = len(positions_2)
                k = j = 0
                while k != len1:
                    while j != len2:
                        if abs(positions_1[k] - positions_2[j]) == 1:
                            query_no = query.split(' ')[0]
                            if (not(query_no, str(i)) in self.boolRes
                                    and int(i)not in doc_keeper):
                                if out == 0:
                                    self.boolRes.append((query_no, str(i)))
                                doc_keeper.append(int(i))
                        elif positions_2[j] > positions_1[k]:
                            break
                        j += 1
                    k += 1
        if (out == 0):
            print(str(len(doc_keeper)) + " results found.")
        return doc_keeper

    def proximitySearch(self, query):
        """
        This function takes care of proximity search where the input looks
        like #20(income, taxes). It updates the boolRes (storing the output
        for boolean searches) list with a tuple of the form (query_no,
        document).

        Args:
            query (String): This function takes the query string as an input
        """
        count = 0
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
        if (term1 in self.inv_index and term2 in self.inv_index):
            # GETTING THE DOC IDS THAT CONTAIN BOTH TERM
            out1 = [doc_id for doc_id in self.inv_index[term1][1]]
            out2 = [doc_id for doc_id in self.inv_index[term2][1]]
            intersec = list(set(out1) & set(out2))
            # IF THE INTERSECTION IS NULL WE STOP THE SEARCH
            if intersec == []:
                pass
            for i in intersec:
                # GETTING THE TERMS POSITIONS FOR EACH DOC
                positions_1 = self.inv_index[term1][1][i]
                positions_2 = self.inv_index[term2][1][i]

                len1 = len(positions_1)
                len2 = len(positions_2)
                k = j = 0
                while k != len1:
                    while j != len2:
                        if abs(positions_1[k] - positions_2[j]) <= proximity:
                            query_no = query.split(' ')[0]
                            if not(query_no, str(i)) in self.boolRes:
                                self.boolRes.append((query_no, str(i)))
                                count += 1
                        elif positions_2[j] > positions_1[k]:
                            break
                        j += 1
                    k += 1
        print(str(count) + " results found.")

    def notOperation(self, term, flag=0):
        """
        This function is a helper function for the boolean search.
        It takes care of the NOT operation.

        Args:
            term ([int]): List of docs containing the negated term
            flag (int, optional): This flag is only equal to 0 if the
            negated term is a phrase. Defaults to 0.

        Returns:
            [type]: [description]
        """
        docs = [int(i.replace("_out.txt", ""))
                for i in os.listdir('output_files')]
        if flag == 0:
            # Single Word Negation:
            # Return all the docs that don't contain the
            # negated term
            return [x for x in docs
                    if x not in self.inv_index[term][1]]
        else:
            # Phrase Search Negation:
            # Return all the docs except those in term
            return [x for x in docs if x not in term]

    def booleanSearch(self, query):
        """
        This function takes care of the boolean Search. It handles
        the scenarios where the query contains a "OR" or "AND" operator.
        Note that  Boolean queries will not contain more than one "AND"
        or "OR" operator at a time. But a mix between phrase query and
        one logical "AND" or "OR" operator can be there . Also "AND"
        or "OR" can be mixed with NOT.

        Args:
            query (String): This function takes the query string as an
            input argument.
        """
        # Case when there's an AND in the query
        if "AND" in query:
            # Initialize the terms dictionary and get rid of the
            # query number
            terms = dict()
            new_query = query[2:].strip()
            # Separate the two terms: LHS and RHS of "AND"
            term1 = re.search('AND (.*)', new_query).group(1).strip()
            term2 = re.search('(.*) AND', new_query).group(1).strip()

            # Initialize documents containing term as empty list
            terms[term1] = []
            terms[term2] = []

            # Looping through the terms
            for term in terms:
                # Checking if the term is negated
                if "NOT" in term:
                    # If the term is negated, check if it's a phrase
                    # Or a single term by first getting rid of the "NOT"
                    not_term = re.search('NOT(.*)', term).group(1).strip()
                    if "\"" in not_term:
                        # If it's a phrase, call phraseSearch first
                        # Then call the notOperation helper with flag
                        not_term = self.phraseSearch(not_term, out=1)
                        terms[term] = self.notOperation(not_term, flag=1)
                    else:
                        # Otherwise, preprocess the term and call
                        # the notOperation helper
                        not_term = self.ps.stem(not_term.lower())
                        terms[term] = self.notOperation(not_term)
                else:
                    # It the term is not negated, check if it's a
                    # phrase or a single term
                    if "\"" in term:
                        # If it's a phrase call phraseSearch
                        terms[term] = self.phraseSearch(term, out=1)
                    else:
                        # Otherwise, preprocess the term and look
                        # for the docs containing it in the inverted
                        # index.
                        tmp_term = self.ps.stem(term.lower())
                        terms[term] = [d for d in self.inv_index[tmp_term][1]]
            # Since we are doing an "AND" operation, we append the
            # intersection of the docs containing term1 and the docs
            # containing term2 to boolRes
            and_list = list(set(terms[term1]) & set(terms[term2]))
            if and_list != []:
                for item in and_list:
                    query_no = query.split(' ')[0]
                    self.boolRes.append((query_no, str(item)))
            print(str(len(and_list)) + " results found.")
        # Case when there's an OR in the query
        elif "OR" in query:
            # Initialize the terms dictionary and get rid of the
            # query number
            terms = dict()
            new_query = query[2:].strip()
            # Separate the two terms: LHS and RHS of "OR"
            term1 = re.search('OR (.*)', new_query).group(1).strip()
            term2 = re.search('(.*) OR', new_query).group(1).strip()

            # Initialize documents containing term as empty list
            terms[term1] = []
            terms[term2] = []

            # Looping through the terms
            for term in terms:
                # Checking if the term is negated
                if "NOT" in term:
                    # If the term is negated, check if it's a phrase
                    # Or a single term by first getting rid of the "NOT"
                    not_term = re.search('NOT(.*)', term).group(1).strip()
                    if "\"" in not_term:
                        # If it's a phrase, call phraseSearch first
                        # Then call the notOperation helper with flag
                        not_term = self.phraseSearch(not_term, out=1)
                        terms[term] = self.notOperation(not_term, flag=1)
                    else:
                        # Otherwise, preprocess the term and call
                        # the notOperation helper
                        not_term = self.ps.stem(not_term.lower())
                        terms[term] = self.notOperation(not_term)
                else:
                    # It the term is not negated, check if it's a
                    # phrase or a single term
                    tmp_term = self.ps.stem(term.lower())
                    if "\"" in term:
                        # If it's a phrase call phraseSearch
                        terms[term] = self.phraseSearch(term, out=1)
                    else:
                        # Otherwise, preprocess the term and look
                        # for the docs containing it in the inverted
                        # index.
                        terms[term] = [d for d in self.inv_index[tmp_term][1]]
            or_list = terms[term1] + list(set(terms[term2])
                                          - set(terms[term1]))
            # Since we are doing an "OR" operation, we append the
            # union of the docs containing term1 and the docs
            # containing term2 to boolRes
            if or_list != []:
                for item in or_list:
                    query_no = query.split(' ')[0]
                    self.boolRes.append((query_no, str(item)))
                    print(str(len(or_list)) + " results found.")

    def booleanQueryFile(self, file="queries.boolean.txt"):
        """
        This function reads the boolean querries file and redirects
        each query to the right function.

        Args:
            file (str, optional): This function takes a fike as an
            argument. Defaults to "queries.boolean.txt".
        """
        f = open(file)
        lines = f.readlines()
        for line in lines:
            if "AND" in line or "OR" in line or "NOT" in line:
                print("Performing Bool Search for query no " + str(line[0:2]))
                self.booleanSearch(line)
            elif "#" in line:
                print("Perofrming Proximity Search: for query no "
                      + str(line[0:2]))
                self.proximitySearch(line)
            elif "\"" in line:
                print("Performing Phrase Search for query no "
                      + str(line[0:2]))
                self.phraseSearch(line)
            # CASE WHERE THE QUERY ONLY CONTAINS A SINGLE WORDa
            elif len(line[2:].split(' ')) == 1:
                print("Performing Single Word Search for query no "
                      + str(line[0:2]))
                try:
                    term = self.ps.stem(line[2:].strip().lower())
                    docs = self.inv_index[term][1]
                    print(str(len(docs)) + " results found.")
                    for d in docs:
                        self.boolRes.append((line[0], str(d)))
                except KeyError:
                    pass

    def writeBooleanToFile(self):
        """
        This function writes the content of boolRes
        to a formatted text file.
        """
        out = open("results.boolean.txt", "w")
        for a, b in self.boolRes:
            out.write(a + "," + b + "\n")
        out.close()

    # I DIDN'T USE THIS FUNCTION, MISSED THE FACT THAT NO NORMALIZATION
    # WAS NEEDED. PLEASE DISREGARD.
    def normalize(self, scores):
        """
        This function normalizes the scores given by the ranked
        search

        Args:
            scores ([float]): It takes the unormalized scores as
            an input.

        Returns:
            [float]: This function returns the normalized scores
            rounded to 4 decimal places.
        """
        return np.around((scores - np.min(scores)) /
                         (np.max(scores) - np.min(scores)), decimals=4)

    def getTF(self, term, document):
        """
        This function is a helper function for the ranked search.
        It computes the frequency of a term in a specific document.

        Args:
            term (string): This function takes a term as a first argument
            document (int): And a document number as a second argument

        Returns:
            int: This function returns the computed term frequency (tf).
        """
        try:
            tf = len(self.inv_index[term][1][document])
        except KeyError:
            tf = 0
        return tf

    def getDF(self, term):
        """
        This function is a helper function for the ranked search.
        It finds the document frequency of a given term.

        Args:
            term (string): This function takes a term as an argument.

        Returns:
            int: This function returns the document frequency of the
            term (df).
        """
        try:
            df = self.inv_index[term][0]
        except KeyError:
            df = 0
        return df

    def tfidfSearch(self, query):
        """
        This function takes care of performing the ranked search.
        The queries have the following format:
        1 Dow Jones industrial average stocks

        Args:
            query (String): This function takes the query string as an
            argument.
        """
        # We start by preprocessing the terms in the query.
        terms = [self.ps.stem(t) for t in
                 re.sub('[^A-Za-z0-9 ]+', '', query[2:]
                        .strip().lower()).split(' ')]
        # Getting all the docs
        docs = [int(i.replace("_out.txt", "")) for i
                in os.listdir('output_files')]
        N = len(docs)

        # Initializing the scores dict
        scores = dict()
        # Looping through the terms
        for t in terms:
            # First step is getting the document frequency
            df = self.getDF(t)
            for doc in docs:
                # Then for each document we look for the
                # term frequency
                tf = self.getTF(t, doc)
                if df > 0 and tf > 0:
                    # If both the df and tf are greater than 0 then we
                    # compute the idf and the score and we add the document
                    # score to the dict
                    idf = math.log10(N/df)
                    try:
                        scores[doc] += (1+math.log10(tf))*idf
                    except KeyError:
                        scores[doc] = (1+math.log10(tf))*idf
                else:
                    # Otherwise we assign 0 to that document.
                    try:
                        scores[doc] += 0
                    except KeyError:
                        scores[doc] = 0

        # We then sort the dict of scores by value in descending order
        scores = dict(sorted(scores.items(), key=lambda item: item[1],
                             reverse=True))
        query_no = query.split(' ')[0]
        # Finally, we make sure we only append the top 150 results to the
        # rankedRes
        self.rankedRes.append((query_no, list(scores.items())[:150]))

    def rankedQueryFile(self, file="queries.ranked.txt"):
        """
        This function reads the ranked search queries file.
        It also writes the contents of rankedRes to a formatted
        text file.

        Args:
            file (str, optional): This function takes a fike as an
            argument. Defaults to "queries.ranked.txt".
        """
        f = open(file)
        lines = f.readlines()
        out = open("results.ranked.txt", "w")
        for line in lines:
            self.tfidfSearch(line)

        for q, res in self.rankedRes:
            for r in res:
                out.write(q + "," + str(r[0]) + "," + str(round(r[1], 4))
                          + "\n")
        out.close()


if __name__ == '__main__':
    try:
        shutil.rmtree("input_files")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    try:
        shutil.rmtree("output_files")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    se = SearchEngine()
    print("Splitting XML file into input files...")
    se.splittingDocs()
    for filename in os.listdir("input_files/"):
        print("Pre processing " + str(filename) + " ...")
        se.preprocessing(filename)
    try:
        shutil.rmtree("input_files")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    print("Pre pocessing completed!")
    print("Generating inverted index")
    se.inverted_index()
    print("Success! Inverted Positional Index Generated!")
    se.writeIndexToFile()
    try:
        shutil.rmtree("output_files")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    print("Running Boolean queries")
    se.booleanQueryFile()
    se.writeBooleanToFile()
    print("Success! Results at: results.boolean.txt")
    print("Running Ranked Retrival queries")
    se.rankedQueryFile()
    print("Success! Results at: results.ranked.txt")
