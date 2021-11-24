import numpy as np


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
        # Computing means and writing results for system.
        line = "{},mean,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n"\
            .format(i, np.mean(precisions), np.mean(recalls),
                    np.mean(r_precisions), np.mean(avgs),
                    np.mean(dcgs_10), np.mean(dcgs_20))
        out.write(line)


if __name__ == '__main__':
    run_eval()
