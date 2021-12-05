import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from tabulate import tabulate
from scipy.sparse import dok_matrix
from sklearn.svm import SVC
from scipy import stats
import math


class Eval:
    def __init__(self, system_number, qrels_file, system_results_file):
        self.system_number = system_number
        self.relevant_docs = self.read_file(qrels_file)
        self.system_results = dict()
        self.system_results_file = system_results_file

    def read_file(cls, qrels_file):
        """This function reads the qrels_file to initialize the relevant docs
        dictionnary

        Args:
            qrels_file (file): file containing the list of relevant documents
            for each of the 10 queries. The format of the file is as follows:
            query_id,doc_id,relevance

        Returns:
            dict: The function returns a dict mapping queries with relevant
            documents and their relevance score
        """
        parsing = dict()
        with open(qrels_file, 'r', encoding="utf-8") as f:
            # Skipping header line
            for line in f.readlines()[1:]:
                line = line.strip().split(",")
                query_no = int(line[0])
                doc_no = int(line[1])
                relevance = int(line[2])
                if query_no in parsing:
                    parsing[query_no].append((doc_no, relevance))
                else:
                    parsing[query_no] = [(doc_no, relevance)]
        return parsing

    def identify_system_results(self):
        """
        This function reads the system_results file and stores the current
        system results in a global variable dictionnary.
        """
        with open(self.system_results_file, 'r', encoding="utf-8") as f:
            # Skipping header line
            for line in f.readlines()[1:]:
                line = line.strip().split(",")
                # Only keep results if it's the current system
                if line[0] == str(self.system_number):
                    query_no = int(line[1])
                    doc_no = int(line[2])
                    score = float(line[4])
                    if query_no in self.system_results:
                        self.system_results[query_no].append((doc_no, score))
                    else:
                        self.system_results[query_no] = [(doc_no, score)]

    def precision_at_k(self, query, k=10):
        """
        This function helps compute the precision at k for a specific query.

        Args:
            query (int): Argument to take the query number
            k (int, optional): rank. Defaults to 10.

        Returns:
            float: Precision at k
        """
        # Get the k-first documents retrieved by the system
        top_k_res = [values[0] for values in self.system_results[query][:k]]
        # Get the list of documents that should've been retrieved
        compare = [values[0] for values in self.relevant_docs[query]]
        count = 0
        # Increase the count only if the document retrieved is in the list
        # of documents that are supposed to be retrieved
        for doc in top_k_res:
            if doc in compare:
                count += 1
        # Return the precision at k
        return count/k

    def recall_at_k(self, query, k=50):
        """
        This function helps compute the recall at k for a specific query.

        Args:
            query (int): Argument to take the query number
            k (int, optional): rank. Defaults to 50.

        Returns:
            float: recall at k
        """
        # Get the k-first documents retrieved by the system
        top_k_res = [values[0] for values in self.system_results[query][:k]]
        # Get the list of documents that should've been retrieved
        compare = [values[0] for values in self.relevant_docs[query]]
        N = len(compare)
        count = 0
        # Increase the count only if the document retrieved is in the list
        # of documents that are supposed to be retrieved
        for doc in top_k_res:
            if doc in compare:
                count += 1
        # Return the recall at k. Note that for recall, we divide by the number
        # of documents that should've been retrieved instead of the rank k.
        return count/N

    def r_precision(self, query):
        """
        This function helps compute the R-precision for a specific query

        Args:
            query (int): Argument to take the query number

        Returns:
            float: r-precision
        """
        # We take the rank to be the number of documents that are supposed
        # to be retrieved for that specific query
        r = len(self.relevant_docs[query])
        # The r-precision is the precision at k where k is the rank we
        # mentionned above
        return self.precision_at_k(query, k=r)

    def avg_precision(self, query):
        """
        This function helps compute the average precision (AP) for a
        specific query

        Args:
            query (int): Argument to take the query number

        Returns:
            float: Average precision
        """
        # Get the list of documents that should've been retrieved and its size
        no_of_relevant = len(self.relevant_docs[query])
        relevant_docs = [values[0] for values in self.relevant_docs[query]]
        # Get the list of documents that have been retrieved by our system
        retrieved = [values[0] for values in self.system_results[query]]
        # Initialize score list
        scores = []
        # start at rank 0
        k = 0
        count = 0
        for doc in retrieved:
            # Increase the rank at each iteration
            k += 1
            if doc in relevant_docs:
                # If the retrieved doc should've been retrieved append
                # the precision at k.
                count += 1
                scores.append(count/k)
        # We divide by the total number of relevant documents because we should
        # consider scores of 0 as well.
        return sum(scores)/no_of_relevant

    def compute_dg(self, scores):
        """
        Helper function for the discounted gain.
            .. math:: \\sum_{i}^{k}\\frac{rel_{i}}{log_{2}(i)}

        Args:
            scores ([floats]): List of system scores

        Returns:
            float: Discounted Gain score
        """
        dcg = []
        for idx, score in scores:
            res = score/np.log2(idx)
            dcg.append(res)
        return sum(dcg)

    def dcg_at_k(self, query, k):
        """
        This function helps computed the discounted cumulative gain
        at k for a specific query.

        Args:
            query (int): Argument that takes the query number
            k (int): rank

        Returns:
            float : Discounted Cumulative Gain
        """
        # Get the first k documents retrived by our system
        retrieved = [values[0] for values in self.system_results[query][:k]]
        # Initialize scores and dcg
        scores = []
        dcg = 0
        # Loop through retrieved documents
        for idx in range(len(retrieved)):
            # Loop through the list of relevant documents for that query
            # i is the document number and v the relevance score
            for i, v in self.relevant_docs[query]:
                # If the retrieved doc is relevant, we append the idx + 1
                # (To start the indexing at 1) and the relevance score and we
                # break the loop for faster results
                if i == retrieved[idx]:
                    scores.append((idx+1, v))
                    break
        # If the scores list isn't empty and starts at one
        # we compute the discounted gain skipping the first value (because of
        # the log) and add the scofe of index 1 to the dcg
        if scores and scores[0][0] == 1:
            dcg += scores[0][1]
            dcg += self.compute_dg(scores[1:])
        # Otherwise, we call the helper function to compute the discounted gain
        # of our entire scores list
        else:
            dcg = self.compute_dg(scores)
        return dcg

    def idcg_at_k(self, query, k):
        """
            This function computed the ideal discounted cumulative gain at k
            for a specific query
        Args:
            query (int): Argument that takes the query number
            k (int): rank

        Returns:
            float: Ideal Discounted Cumulative gain
        """
        # Get the scores for all the first k relevant docs and index them
        # starting at 1
        scores = [values[1] for values in self.relevant_docs[query][:k]]
        scores = [(idx+1, value) for idx, value in enumerate(scores)]
        # Call the helper function to compute the Discounted Gain for the
        # Scores skipping the first value (because of log) and add score
        # of the first value to the result.
        return scores[0][1] + self.compute_dg(scores[1:])

    def ndcg_at_k(self, query, k):
        """
        This function computed the Normalized discounted cumulative gain at k
        for a specific query
        Args:
            query (int): Argument that takes the query number
            k (int): rank

        Returns:
            float : Normalized Discounted Cumulative Gain.
                    nDCG@k = DCG@k / iDCG@k
        """
        return self.dcg_at_k(query, k=k)/self.idcg_at_k(query, k=k)


def run_eval():
    """ This function uses all the functions defined in the Eval class to generate
    our ir_eval.csv file output """

    out = open("ir_eval.csv", "w")
    # Adding Header
    out.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n")
    Means = {'Precisions': [], 'Recalls': [], 'R_precisions': [], 'APs': [],
                 'DCGS10': [], 'DCGS20': []}
    # Looping through the 6 systems
    for i in range(1, 7):
        precisions = []; recalls = []; r_precisions = []; avgs = []
        dcgs_10 = []; dcgs_20 = []
        e = Eval(i, "qrels.csv", "system_results.csv")
        e.identify_system_results()
        # Looping through the 10 queries
        for j in range(1, 11):
            # Precision at k computing and appending for average
            p_at_k = e.precision_at_k(j); precisions.append(p_at_k)
            # Recall at k computing and appending for average
            r_at_k = e.recall_at_k(j); recalls.append(r_at_k)
            # R-precision computing and appending for average
            r_prec = e.r_precision(j); r_precisions.append(r_prec)
            # Average precision computing and appending for average
            avg_prec = e.avg_precision(j); avgs.append(avg_prec)
            # Normalized discounted gains at k = 10 and k =20 computing
            # and appending for average
            dcg_10 = e.ndcg_at_k(j, 10); dcgs_10.append(dcg_10)
            dcg_20 = e.ndcg_at_k(j, 20); dcgs_20.append(dcg_20)
            # Writing result for system and query
            line = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n"\
                .format(i, j, p_at_k, r_at_k, r_prec, avg_prec, dcg_10, dcg_20)
            out.write(line)
        # Computing means and writing results for system
        line = "{},mean,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n"\
            .format(i, np.mean(precisions), np.mean(recalls),
                    np.mean(r_precisions), np.mean(avgs),
                    np.mean(dcgs_10), np.mean(dcgs_20))
        out.write(line)

        # Appending values for 2 tailed t-test.
        Means['APs'].append(avgs)
        Means['DCGS10'].append(dcgs_10)
        Means['DCGS20'].append(dcgs_20)
        Means['R_precisions'].append(r_precisions)
        Means['Recalls'].append(recalls)
        Means['Precisions'].append(precisions)

    out.close()

    # Writing 2-tailed t-test outputs to file for report.
    with open("2-tailed t-test.txt", "w") as out:
        rows, cols = (6, 6)
        for metric in Means.keys():
            out.write("\n" + metric + "\n")
            arr = [[0]*cols]*rows
            system_scores = Means[metric]
            for i in range(6):
                for j in range(6):
                    res = list(stats.ttest_ind(system_scores[i],
                                                system_scores[j]))
                    out.write("({:.3f},{:.3f}) ".format(res[0], res[1]))
                    arr[i][j] = (res[0], res[1])
                out.write("\n")


# ######################## PART 2 ########################
def preprocessing(text):
    """
    This function takes care of the preprocessing of the inputted
    text. It does case folding, stopping, stemming and tokenization

    Args:
        text (String): Text to pre-process

    Returns:
        [String]: String list of the preprocessed input        
    """
    # Get stop words from file
    stop_words = open("englishST.txt", "r")
    st_words = [word.strip() for word in stop_words.readlines()]
    stop_words.close()

    # Initializing stemmer (from nltk library)
    ps = SnowballStemmer(language='english')

    # Keeping words only and splitting at spaces
    keep_words = re.sub(r"[\W]", " ", text)
    tokens = keep_words.split()

    # Stemming, case-folding, stopping and tokenizing
    res = [ps.stem(i.lower()) for i in tokens if i.lower() not in st_words]

    return res


def get_freq(corpus_df):
    """
    This function counts in how many documents each unique term
    of the inputted corpus appears

    Args:
        corpus_df (Pandas DataFrame): Corpus DataFrame

    Returns:
        dict(): Dictionnary with key: word and value: number of documents
        it appears in.
    """
    # Initializing the dict
    corpus_freq = dict()
    # Getting content of column 1(only text, not corpus name)
    text = [corpus_df[1][i]+"\n" for i in range(corpus_df.shape[0])]
    # Looping through each document (line in text)
    for i in text:
        # Looping through each unique term in the document
        for j in set(preprocessing(i)):
            # Updating the frequency
            if j in corpus_freq:
                corpus_freq[j] += 1
            else:
                corpus_freq[j] = 1
    return corpus_freq


def index_frequency(file="train_and_dev.tsv"):
    """
    This function generates an index dictionnary for each corpus from
    the tsv file.

    Args:
        file (str, optional): File storing our data.
        Defaults to "train_and_dev.tsv".

    Returns:
        [String], dict(): Returns a list containing all the unique terms
        and a dict storing the index for each of the corpus.
        The index itself is a dictionnary where each term is mapped to the
        number of documents it appears in.
    """
    df = pd.read_csv(file, delimiter=r"\t", header=None, engine='python')
    text = "".join([df[1][i]+"\n" for i in range(df.shape[0])])
    text = set(preprocessing(text))

    quran = df.where(df[0] == 'Quran').dropna().reset_index(drop=True)
    nt = df.where(df[0] == 'NT').dropna().reset_index(drop=True)
    ot = df.where(df[0] == 'OT').dropna().reset_index(drop=True)
    lengths = {'Quran': len(quran), 'NT': len(nt), 'OT': len(ot)}

    quran = get_freq(quran)
    nt = get_freq(nt)
    ot = get_freq(ot)

    return text, {'Quran': quran, 'NT': nt, 'OT': ot}, lengths


def MI(N00, N01, N10, N11):
    """
    This function computes the Mutual Information for a term.
    Following the formula given in the slides.

    Args:
        N00 (int): length of target corpus - N10
        N01 (int): sum of lengths of the 2 other corpora - N11
        N10 (int): number of documents the term appeared in the other
        2 corpora
        N11 (int): number of documents the term appeared in the target
        corpus

    Returns:
        float: Mutual information for the term
    """
    N = N00 + N01 + N10 + N11
    l1 = N11 + N10
    l0 = N01 + N00
    r1 = N01 + N11
    r0 = N00 + N10
    P1 = np.log2(N*N11 / (l1*r1)) if N*N11 != 0 and l1*r1 != 0 else 0
    P2 = np.log2(N*N01 / (l0*r1)) if N*N01 != 0 and l0*r1 != 0 else 0
    P3 = np.log2(N*N10 / (l1*r0)) if N*N10 != 0 and l1*r0 != 0 else 0
    P4 = np.log2(N*N00 / (l0*r0)) if N*N00 != 0 and l0*r0 != 0 else 0
    return (N11/N)*P1 + (N01/N)*P2 + (N10/N)*P3 + (N00/N)*P4


def CHI(N00, N01, N10, N11):
    """
    This function computes the Chi squared score for a term.
    Following the formula given in the slides.

    Args:
        N00 (int): length of target corpus - N10
        N01 (int): sum of lengths of the 2 other corpora - N11
        N10 (int): number of documents the term appeared in the other
        2 corpora
        N11 (int): number of documents the term appeared in the target
        corpus

    Returns:
        float: Chi Squared score for the term
    """
    N = N00 + N01 + N10 + N11
    return (N * ((N11 * N00 - N10 * N01) ** 2)) / ((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))


def MI_X2_Res(text, freqs, lengths):
    """
    This function generates all the Mutual Information and Chi Squared
    scores for all the corpuses.

    Args:
        text ([String]): list containing all the unique terms from all
        the corpora
        freqs (dict): dict storing the frequency index for each of the corpus.
        The index itself is a dictionnary where each term is mapped to the
        number of documents it appears in.
        lengths (dict): dict storing the lengths of all the corpora

    Returns:
        MIs (dict()): A dictionnary with the Mutual Information scores
        for all the corpora.
        Chis (dict()): A dictionnary with the Chi Squared scores for
        all the corpora.
    """
    MIs = dict()
    Chis = dict()
    texts = ['Quran', 'NT', 'OT']
    # Looping through corpora
    for corpus in texts:
        MIs[corpus] = dict()
        Chis[corpus] = dict()
        # Get the other 2 corpora
        compare = [c for c in texts if c != corpus]
        # Get the lengths
        target_length = lengths[corpus]
        comapre_length = sum([lengths[c] for c in compare])
        # Looping through every unique term
        for term in text:
            # Get N11: number of documents the term appeared in
            # the target corpus using our freqs dict
            if term in freqs[corpus]:
                N11 = freqs[corpus][term]
            else:
                N11 = 0
            # Compute N01 from N11
            N01 = target_length - N11

            # Get N10: number of documents the term appeared in
            # the other 2 corpora using our freqs dict
            N10 = 0
            for doc in compare:
                if term in freqs[doc]:
                    N10 += freqs[doc][term]
            # Compute N00 from N11
            N00 = comapre_length - N10

            # Using the 4 computed values call the MI and CHI helper
            # functions and update the output dictionnaries.
            MIs[corpus][term] = MI(N00, N01, N10, N11)
            Chis[corpus][term] = CHI(N00, N01, N10, N11)

    return MIs, Chis


def generate_ranked_list(MIs, Chis):
    """
    This function sorts dictionnaries and keeps the top 10 results.
    We use it for the Mutual Information and Chi Squared scores.
    Args:
        MIs (dict): Dictionnary storing all the Mutual Information
        scores for each corpus
        Chis (dict):  Dictionnary storing all the Chi Squared scores
        for each corpus

    Returns:
        results_MI (dict): Top 10 terms with highest Mutual Information
        scores for each corpus
        results_Chi (dict): Top 10 terms with highest Chi Squared scores
        for each corpus
    """
    results_MI = dict()
    results_Chi = dict()
    texts = ['Quran', 'NT', 'OT']
    for corpus in texts:
        results_MI[corpus] = list(dict(sorted(
            MIs[corpus].items(), key=lambda item: item[1], reverse=True))
                                  .items())[:10]
        results_Chi[corpus] = list(dict(sorted(
            Chis[corpus].items(), key=lambda item: item[1], reverse=True))
                                   .items())[:10]
    return results_MI, results_Chi


def write_ranked(ranked_m, ranked_c):
    """
    Function that writes the Mutual Information and Chi Squared scores
    to a file.
    Args:
        ranked_m (dict): Top 10 terms with highest Mutual Information
        scores for each corpus
        ranked_c (dict): Top 10 terms with highest Chi Squared scores
        for each corpus
    """
    out = open("ranked_results.txt", "w")
    out.write("Mutual Information\n")
    out.write(tabulate(ranked_m.items()))
    out.write("\nChi Squared\n")
    out.write(tabulate(ranked_c.items()))
    out.close()


def preprocessing_no_stem(text):
    """
    This function takes care of the preprocessing of the inputted
    text without the stemming. It does case folding, stopping,
    and tokenization.
    Args:
        text (String): Text to pre-process

    Returns:
        [String]: String list of the preprocessed input.
    """
    # Stop words preprocessing
    stop_words = open("englishST.txt", "r")
    st_words = [word.strip() for word in stop_words.readlines()]
    stop_words.close()

    keep_words = re.sub(r"[\W]", " ", text)
    tokens = keep_words.split()
    res = [i.lower() for i in tokens if i.lower() not in st_words]

    return res


def get_verses(file="train_and_dev.tsv"):
    """
    This function creates a dictionnary mapping each corpus to a
    list of lists. Where each element in the list is a preprocessed
    line of the corpus.
    This is useful for LDA.
    Args:
        file (str, optional):  File storing our data.
        Defaults to "train_and_dev.tsv".

    Returns:
        N (int): length of all the corpora combined
        verses (dict): dictionnary mapping each corpus to a list
        of lists.
    """
    verses = dict()

    df = pd.read_csv(file, delimiter=r"\t", header=None, engine='python')
    verses['All'] = [preprocessing_no_stem(i) for i in df[1]]

    quran = df.where(df[0] == 'Quran').dropna().reset_index(drop=True)
    nt = df.where(df[0] == 'NT').dropna().reset_index(drop=True)
    ot = df.where(df[0] == 'OT').dropna().reset_index(drop=True)

    verses['Quran'] = [preprocessing(i) for i in quran[1]]
    verses['NT'] = [preprocessing(i) for i in nt[1]]
    verses['OT'] = [preprocessing(i) for i in ot[1]]

    return len(verses['All']), verses


def get_average_score(scores, i, j):
    """
    Helper function for getLDA(verses, length)
    Gets the average score from a slice of a score list.
    Args:
        scores ([(topic, score)]): List of Topic distribution for the whole document.
        Each element in the list is a pair of a topic’s id, and the probability
        that was assigned to it.
        i (int): slice start
        j (int): slice end

    Returns:
        float: Average score.
    """
    score = [0]*20
    for doc in scores[i:j]:
        for item in doc:
            score[item[0]] += item[1] / len(scores)
    return score


def get_LDA(verses, lengths):
    """
    This function takes care of the following tasks:
    - For each corpus, computes the average score for each topic by
    summing the document-topic probability for each document in that
    corpus and dividing by the total number of documents in the corpus.
    - Then, for each corpus, indetifies the topic that has the highest 
    average score (3 topics in total). For each of those three topics
    find the top 10 tokens with highest probability of belonging to that topic.

    Args:
        verses (dict): dictionnary mapping each corpus to a list
        of lists.
        lengths (dict):  dict storing the lengths of all the corpora

    Returns:
        top_words (dict): dict storing the top 10 tokents with highest
        probability of belonging to top topic for each corpus
        top_topics (dict): dict storing the top topic for each corpus
    """
    all_text = verses['All']
    scores = []

    # Initialize LDA model
    dictionary = Dictionary(all_text)
    corpus = [dictionary.doc2bow(text) for text in all_text]
    model = LdaModel(corpus, id2word=dictionary, num_topics=20)

    # Get all the topic distribution scores
    for i in range(len(all_text)):
        scores.append(model.get_document_topics(corpus[i]))

    # Get the indices for slicing ready based on order in tsv file
    slice_OT = lengths['OT']
    slice_NT = lengths['OT']+lengths['NT']
    slice_Quran = sum([lengths[i] for i in lengths.keys()])
    avgs = dict()

    # Get the average scores per corpus using helper function defined
    # above
    avgs['OT'] = get_average_score(scores, 0, slice_OT)
    avgs['NT'] = get_average_score(scores, slice_OT, slice_NT)
    avgs['Quran'] = get_average_score(scores, slice_NT, slice_Quran)

    # Get the top topic (with highest score for each corpus)
    top_topics = dict()
    for corpus in avgs.keys():
        top_topics[corpus] = avgs[corpus].index(max(avgs[corpus]))

    # For each corpus, get the top topic and find the top 10 tokens with
    # highest probability of belonging to that topic
    top_words = dict()
    for corpus in top_topics.keys():
        topic_to_word = dict()
        topic = top_topics[corpus]
        topic_to_word[topic] = model.show_topic(topic)
        top_words[corpus] = topic_to_word

    return top_words, top_topics


def write_LDA_results(top_topics, top_words):
    """
    Writing the LDA_results found to a file.
    Args:
        top_topics (dict): dict storing the top topic for each corpus
        top_words (dict): dict storing the top 10 tokents with highest
        probability of belonging to top topic for each corpus
    """
    corpora = ['Quran', 'OT', 'NT']
    with open("LDA_results.txt", "w") as out:
        for corpus in corpora:
            out.write(str(corpus))
            topic = top_topics[corpus]
            out.write("\n")
            out.write(str(topic) + ": ")
            out.write(", ".join(["(%s,%s)" % i
                                 for i in top_words[corpus][topic]]))
            out.write("\n\n")


# ##################### PART 3 #####################
def tokenize(text):
    """
    This function only tokenizes the inputted text.
    Only gets rid of punctuation.
    Args:
        text (String): Text to pre-process

    Returns:
        [String]: String list of the tokinzed input
    """
    keep_words = re.sub(r"[\W]", " ", text)
    tokens = keep_words.split()
    return [i.lower() for i in tokens]


def dataset_splitting(file="train_and_dev.tsv", n=90, test="test.tsv"):
    """
    This function shuffles the order of the data and splits the dataset into
    a training set and a separate development set.
    It also loads the test set from the given test file
    Args:
        file (str, optional): File containing training and development data.
        Defaults to "train_and_dev.tsv".
        n (int, optional): Where we split for testing. Defaults to 90.
        test (str, optional):File containing test data. Defaults to "test.tsv".

    Returns:
        d (dict): mapping from the categories (corpora names) to numberic IDs
        train_df (Pandas dataFrame): Training dataFrame
        dev_df (Pandas dataFrame): Development dataFrame
        test_df (Pandas dataFrame): Testing dataFrame
        df (Pandas dataFrame): Shuffled original dataFrame
    """

    # Loading the original dataFrame and shuffling
    df = pd.read_csv(file, delimiter=r"\t", header=None, engine='python')
    df = df.sample(frac=1, random_state=100)

    # Creating mapping from the categories (corpora names) to numberic IDs
    categories = [i for i in df[0]]
    d = dict([(y, x+1) for x, y in enumerate(sorted(set(categories)))])

    # Getting splitting index to separate train and dec
    split = int(len(df)*(n/100))

    # Creating training and development dataFrames
    train_df = df[:split].reset_index(drop=True)
    dev_df = df[split:].reset_index(drop=True)

    # Loading test dataFrame from file
    test_df = pd.read_csv(test, delimiter=r"\t", header=None, engine='python')
    return d, train_df, dev_df, test_df, df


def ID_mapping(train_df, dev_df, test_df, improved=False):
    """
    This function finds all the unique terms, and give each of them a unique ID
    (starting from 0 to the number of terms).

    Args:
        train_df (Pandas dataFrame): Training dataFrame
        dev_df (Pandas dataFrame): Development dataFrame
        test_df (Pandas dataFrame): Test dataFrame
        improved (bool, optional): If it's true, different way to process the frames.
        Defaults to False (for baseline model).

    Returns:
        no_of_terms (int): total number of terms
        ID_map_train (dict): mapping of unique IDs for each training term
        ID_map_dev (dict): mapping of unique IDs for each development term
        ID_map_test (dict): mapping of unique IDs for each test term
    """

    ID_map_train = dict()
    text_train = "".join([train_df[1][i]+"\n" for i in range(train_df.shape[0])])

    ID_map_dev = dict()
    text_dev = "".join([dev_df[1][i]+"\n" for i in range(dev_df.shape[0])])

    ID_map_test = dict()
    text_test = "".join([test_df[1][i]+"\n" for i in range(test_df.shape[0])])

    # Preprocessing text depending on which model we are training.
    if improved:
        text_train = set(preprocessing(text_train))
        text_dev = set(preprocessing(text_dev))
        text_test = set(preprocessing(text_test))
    else:
        text_train = set(tokenize(text_train))
        text_dev = set(tokenize(text_dev))
        text_test = set(tokenize(text_test))

    # Assigning unique IDs to terms while maintaining consistency
    # over the corpora.
    count = 0
    for word in text_train:
        ID_map_train[word] = count
        count += 1

    for word in text_dev:
        if word in ID_map_train:
            ID_map_dev[word] = ID_map_train[word]
        else:
            ID_map_dev[word] = count
            print(word)
            count += 1

    for word in text_test:
        if word in ID_map_train:
            ID_map_test[word] = ID_map_train[word]
        elif word in ID_map_dev:
            ID_map_test[word] = ID_map_dev[word]
        else:
            ID_map_test[word] = count
            count += 1

    # Getting total number of terms
    all_terms = text_train.union(text_test).union(text_dev)
    no_of_terms = len(all_terms)

    return no_of_terms, ID_map_train, ID_map_dev, ID_map_test


def tf_idf(doc_freq, count, word, N):
    """
    Helper function that computes the tf-idf score for a term.

    Args:
        doc_freq (dict): dictionnary mapping a term with its
        document frequency
        count (int): Term frequency
        word (String): Term considered
        N (int): Total number of documents

    Returns:
        tf*idf (float): tf-idf score for the considered term.
    """
    try:
        df = doc_freq[word]
    except KeyError:
        df = 0
    tf = (1+math.log10(count))
    try:
        idf = math.log10(N/df)
    except ZeroDivisionError:
        idf = 0
    return tf * idf


def generate_matrix(ID_map, df, no_of_terms, d, doc_freq ={}, N=0, improved=False):
    """
    This function creates a count matrix where each row is a document and
    each column corresponds to a word in the vocabulary.
    If we are generating for the baseline model then (improved = False) then
    the value of element (i,j) is the count of the number of times the word
    j appears in document i.
    Otherwise: The value of element (i,j) is the tf-idf score of the word j
    for document i.

    Args:
        ID_map (dict): mapping of unique IDs for each training term
        df (Pandas dataFrame): training, dev or test dataFrame
        no_of_terms (int): total number of terms in the corpora
        d (dict): mapping from the categories (corpora names) to numberic IDs
        improved (bool, optional): If improved different processing and scoring
        function. Defaults to False.
        N (int): total number of documents (used for tf-idf)

    Returns:
        cats ([int]): true labels (y-input)
        S (Sparse Matrix): X input for classifier
    """
    # Pre processing and getting the verses differently
    # depending on which model we are training
    # if improved:
    #     verses = [preprocessing(i) for i in df[1]]
    # else:
    verses = [tokenize(i) for i in df[1]]

    # Getting true y-labels
    categories = [i for i in df[0]]
    cats = [d[x] for x in categories]

    # Initializing sparse matrix
    S = dok_matrix((len(verses), no_of_terms))

    # Updating sparse matrix with score
    for i in range(len(verses)):
        count_dict = {t: verses[i].count(t) for t in verses[i]}
        for item in count_dict.keys():
            word = item
            count = int(count_dict[item])
            word_idx = ID_map[word]
            # Different score function depending on model trained
            if improved:
                S[i, word_idx] = tf_idf(doc_freq, count, word, N) + count
            else:
                S[i, word_idx] = count

    return cats, S


def SVM_model(X_train, X_dev, X_test, y_train, c=1000):
    """
    This function takes care of the classifier part.
    Training on training set and predicting test and development
    labels.

    Args:
        X_train (Sparse Matrix): X train inputs
        X_dev (Sparse Matrix): X development inputs
        X_test (Sparse Matrix): X test inputs
        y_train ([int]): true labels for training set
        c (int, optional): c parameter for SVM. Defaults to 1000.

    Returns:
        y_pred_dev (dict): Predicted labels for development set
        y_pred_train (dict): Predicted labels for training set
        y_pred_test (dict): Predicted labels for test set
    """
    model = SVC(C=c)
    # Training
    model.fit(X_train, y_train)
    # Predicting
    y_pred_dev = model.predict(X_dev)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_dev, y_pred_train, y_pred_test


def scores(y_pred, y_true, d):
    """
    This function computes all the required scores: precision, recall, and f1-score
    for each of the 3 classes, as well as the macro-averaged precision, recall, and
    f1-score across all three classes.

    Args:
        y_pred ([int]): List of predicted outputs
        y_true ([type]): List of true labels
        d (dict): mapping from the categories (corpora names) to numberic IDs

    Returns:
        precisions (dict): Dictionnary storing the precision scores for each of the
        3 classes, as well as the macro-averaged precision.
        recalls (dict): Dictionnary storing the recall scores for each of the 3 classes,
        as well as the macro-averaged recall.
        f1_scores (dict): Dictionnary storing the f1-scores for each of the 3 classes,
        as well as the macro-averaged f1-scores.
    """
    classes = np.unique(np.append(y_pred, y_true))

    # Making dataframes out of the prediction and true labels
    dfpred = pd.DataFrame(y_pred)
    dftrue = pd.DataFrame(y_true)

    # Initializing scores dictionnaries
    precisions = dict()
    recalls = dict()
    f1_scores = dict()

    # Creating a mapping from the numberic IDs to the categories (corpora names)
    rev_d = dict()
    for item in d.keys():
        rev_d[d[item]] = item

    # Looping through all the classes
    for c in classes:
        idx = rev_d[c]
        pred = dfpred[dfpred[0] == c]
        index_pred = pred.index.tolist()
        true = dftrue[dftrue[0] == c]
        index_true = dftrue.reindex(index=index_pred)

        # Computing precision, recall and f1-score and updating the
        # dictionnary
        p = sum(np.array(pred) == np.array(index_true)) / len(pred)
        r = sum(np.array(pred) == np.array(index_true)) / len(true)
        f = 2*p*r / (p+r)

        precisions[idx] = float(p[0])
        recalls[idx] = float(r[0])
        f1_scores[idx] = float(f[0])

    # Getting Macro values and updating the dictionnary
    macro_p = np.mean(list(precisions.values()))
    macro_r = np.mean(list(recalls.values()))
    macro_f = 2*macro_p*macro_r / (macro_p+macro_r)

    precisions['Macro'] = float(macro_p)
    recalls['Macro'] = float(macro_r)
    f1_scores['Macro'] = float(macro_f)

    return precisions, recalls, f1_scores


def get_freq_improved(corpus_df):
    corpus_freq = dict()
    text = [corpus_df[1][i]+"\n" for i in range(corpus_df.shape[0])]
    for i in text:
        for j in tokenize(i):
            if j in corpus_freq:
                corpus_freq[j] += 1
            else:
                corpus_freq[j] = 1
    return corpus_freq


def write_classification_baseline(y_train, y_dev, y_test, y_pred_dev,
                                  y_pred_train, y_pred_test, d):
    """
    Function that writes the scores for the baseline algorithm to a file.
    Args:
        y_train ([int]): True labels for training set
        y_dev ([int]): True labels for development set
        y_test ([int]): True labels for test set
        y_pred_dev ([int]): Predicted development labels
        y_pred_train ([int]): Predicted train labels
        y_pred_test ([int]): Predicted test labels
        d (dict): mapping from the categories (corpora names) to numberic IDs
    """
    with open("classification.csv", "w") as out:
        out.write("system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro")
        out.write("\n")
        precisions, recalls, f1_scores = scores(y_pred_train, y_train, d)
        out.write("baseline,train,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}"\
            .format(precisions['Quran'], recalls['Quran'], f1_scores['Quran'], precisions['OT'], recalls['OT'], f1_scores['OT'],\
                precisions['NT'], recalls['NT'], f1_scores['NT'], precisions['Macro'], recalls['Macro'], f1_scores['Macro']))
        out.write("\n")
        precisions, recalls, f1_scores = scores(y_pred_dev, y_dev, d)
        out.write("baseline,dev,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}"\
            .format(precisions['Quran'], recalls['Quran'], f1_scores['Quran'], precisions['OT'], recalls['OT'], f1_scores['OT'],\
                precisions['NT'], recalls['NT'], f1_scores['NT'], precisions['Macro'], recalls['Macro'], f1_scores['Macro']))
        out.write("\n")
        precisions, recalls, f1_scores = scores(y_pred_test, y_test, d)
        out.write("baseline,test,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}"\
            .format(precisions['Quran'], recalls['Quran'], f1_scores['Quran'], precisions['OT'], recalls['OT'], f1_scores['OT'],\
                precisions['NT'], recalls['NT'], f1_scores['NT'], precisions['Macro'], recalls['Macro'], f1_scores['Macro']))
        out.write("\n")


def write_classification_improved(y_train, y_dev, y_test, y_pred_dev,
                                  y_pred_train, y_pred_test, d):
    """
    Function that appends the results of the improved algorithm to the file containing
    the scores for the baseline model.

    Args:
        y_train ([int]): True labels for training set
        y_dev ([int]): True labels for development set
        y_test ([int]): True labels for test set
        y_pred_dev ([int]): Predicted development labels
        y_pred_train ([int]): Predicted train labels
        y_pred_test ([int]): Predicted test labels
        d (dict): mapping from the categories (corpora names) to numberic IDs
    """
    with open("classification.csv", "a") as out:
        precisions, recalls, f1_scores = scores(y_pred_train, y_train, d)
        out.write("improved,train,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}"\
            .format(precisions['Quran'], recalls['Quran'], f1_scores['Quran'], precisions['OT'], recalls['OT'], f1_scores['OT'],\
                precisions['NT'], recalls['NT'], f1_scores['NT'], precisions['Macro'], recalls['Macro'], f1_scores['Macro']))
        out.write("\n")
        precisions, recalls, f1_scores = scores(y_pred_dev, y_dev, d)
        out.write("improved,dev,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}"\
            .format(precisions['Quran'], recalls['Quran'], f1_scores['Quran'], precisions['OT'], recalls['OT'], f1_scores['OT'],\
                precisions['NT'], recalls['NT'], f1_scores['NT'], precisions['Macro'], recalls['Macro'], f1_scores['Macro']))
        out.write("\n")
        precisions, recalls, f1_scores = scores(y_pred_test, y_test, d)
        out.write("improved,test,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}"\
            .format(precisions['Quran'], recalls['Quran'], f1_scores['Quran'], precisions['OT'], recalls['OT'], f1_scores['OT'],\
                precisions['NT'], recalls['NT'], f1_scores['NT'], precisions['Macro'], recalls['Macro'], f1_scores['Macro']))
        out.write("\n")


if __name__ == '__main__':
    """
    Runs the entire system
    """
    # TASK 1
    run_eval()

    # TASK 2
    total, freqs, lengths = index_frequency()
    
    outss = ['yet', 'i', 'will', 'also', 'make', 'a', 'nation', 'of', 'the', 'son', 'of', 'the', 'bondwoman', 'because', 'he', 'is', 'your', 'seed', 
             'so', 'i', 'swore', 'in', 'my', 'wrath', 'they', 'shall', 'not', 'enter', 'my', 'rest',
             'pursue', 'peace', 'with', 'all', 'people', 'and', 'holiness', 'without', 'which', 'no', 'one', 'will', 'see', 'the', 'lord',
             'and', 'they', 'let', 'go', 'the', 'anchors', 'and', 'left', 'them', 'in', 'the', 'sea', 'meanwhile', 'loosing', 'the', 'rudder', 'ropes', 'and', 'they', 'hoisted', 'the', 'mainsail', 'to', 'the', 'wind', 'and', 'made', 'for', 'shore',
             'when', 'moses', 'saw', 'it', 'he', 'marveled', 'at', 'the', 'sight', 'and', 'as', 'he', 'drew', 'near', 'to', 'observe', 'the', 'voice', 'of', 'the', 'lord', 'came', 'to', 'him',
             'and', 'he', 'turned', 'to', 'their', 'idols', 'and', 'asked', 'them', 'do', 'you', 'eat']
    
    ps = SnowballStemmer(language='english')
    with open("temp.txt", "w") as t:
        t.write("word, OT, NT, Quran\n")
        for word in outss:
            word = ps.stem(word)
            try:
                freq_OT = freqs['OT'][word]
            except KeyError:
                freq_OT = 0
            try:
                freq_NT = freqs['NT'][word]
            except KeyError:
                freq_NT = 0
            try:
                freq_Quran = freqs['Quran'][word]
            except KeyError:
                freq_Quran = 0
            t.write('{},{},{},{}\n'.format(word, freq_OT, freq_NT, freq_Quran))
    
    MIs, Chis = MI_X2_Res(total, freqs, lengths)
    ranked_m, ranked_c = generate_ranked_list(MIs, Chis)
    write_ranked(ranked_m, ranked_c)
    N, verses = get_verses()
    top_words, top_topics = get_LDA(verses, lengths)
    write_LDA_results(top_topics, top_words)

    # TASK 3 SPLITTING + MAPPING
    d, train_df, dev_df, test_df, original_df = dataset_splitting()

    # TASK 3 BASELINE
    no_of_terms, ID_map_train, ID_map_dev, ID_map_test = ID_mapping(train_df, dev_df, test_df)
    y_train, X_train = generate_matrix(ID_map_train, train_df, no_of_terms, d)
    y_dev, X_dev = generate_matrix(ID_map_dev, dev_df, no_of_terms, d)
    y_test, X_test = generate_matrix(ID_map_test, test_df, no_of_terms, d)
    y_pred_dev, y_pred_train, y_pred_test = SVM_model(X_train, X_dev, X_test, y_train)
    breakpoint = 0
    
    # Get some Misclassified examples from development set
    print(d)
    verses_copy = [tokenize(i) for i in dev_df[1]]
    for i, (a, b) in enumerate(zip(y_dev, y_pred_dev)):
        if breakpoint == 8:
            break
        if a != b:
            print('Actual Label: ' + str(a) + "\n")
            print('Predicted Label: ' + str(b) + "\n")
            print('Verse: ')
            print(verses_copy[i])
            print("\n")
            breakpoint += 1

    write_classification_baseline(y_train, y_dev, y_test,
                                  y_pred_dev, y_pred_train, y_pred_test, d)

    # TASK 3 IMPROVED MODEL
    doc_freq = get_freq_improved(original_df)
    no_of_terms, ID_map_train, ID_map_dev, ID_map_test = ID_mapping(train_df, dev_df, test_df)
    y_train, X_train = generate_matrix(ID_map_train, train_df, no_of_terms, d,
                                       doc_freq, N, improved=True)
    y_dev, X_dev = generate_matrix(ID_map_dev, dev_df, no_of_terms, d, doc_freq,
                                   N, improved=True)
    y_test, X_test = generate_matrix(ID_map_test, test_df, no_of_terms, d, doc_freq,
                                     N, improved=True)
    y_pred_dev, y_pred_train, y_pred_test = SVM_model(X_train, X_dev, X_test, y_train, c=40)
    write_classification_improved(y_train, y_dev, y_test,
                                  y_pred_dev, y_pred_train, y_pred_test, d)
