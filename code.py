
class Eval:
    def __init__(self, system_number, qrels_file, system_results_file):
        self.system_number = system_number
        self.relevant_docs = self.read_file(qrels_file)
        self.system_results = dict()
        self.system_results_file = system_results_file

    def read_file(cls, qrels_file):
        parsing = dict()
        with open(qrels_file, 'r', encoding="utf-8") as f:
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
        with open(self.system_results_file, 'r', encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                line = line.strip().split(",")
                if line[0] == str(self.system_number):
                    query_no = int(line[1])
                    doc_no = int(line[2])
                    score = float(line[4])
                    if query_no in self.system_results:
                        self.system_results[query_no].append((doc_no, score))
                    else:
                        self.system_results[query_no] = [(doc_no, score)]

    def precision_at_k(self, query, k=10):
        top_k_res = [values[0] for values in self.system_results[query][:k]]
        compare = [values[0] for values in self.relevant_docs[query]]
        count = 0
        for doc in top_k_res:
            if doc in compare:
                count += 1
        return count/k

    def recall_at_k(self, query, k=50):
        top_k_res = [values[0] for values in self.system_results[query][:k]]
        compare = [values[0] for values in self.relevant_docs[query]]
        N = len(compare)
        count = 0
        for doc in top_k_res:
            if doc in compare:
                count += 1
        return count/N

    def r_precision(self, query):
        r = len(self.relevant_docs[query])
        return self.precision_at_k(query, k=r)

    def avg_precision(self, query):
        no_of_relevant = len(self.relevant_docs[query])
        relevant_docs = [values[0] for values in self.relevant_docs[query]]
        retrieved = [values[0] for values in self.system_results[query]]
        scores = []
        k = 0
        count = 0
        for doc in retrieved:
            k += 1
            if doc in relevant_docs:
                count += 1
                scores.append(count/k)
        return sum(scores)/no_of_relevant

    def dcg_at_k(self, query, k):
        pass

    def idcg_at_k(self, query, k):
        pass

    def ndcg_at_k(self, query, k):
        pass


if __name__ == '__main__':
    out = open("ir_eval.csv", "w")
    out.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n")
    for i in range(1, 7):
        precisions = []
        recalls = []
        r_precisions = []
        avgs = []
        e = Eval(i, "qrels.csv", "system_results.csv")
        e.identify_system_results()
        for j in range(1, 11):
            out.write(str(i) + "," + str(j) + ",")
            p_at_k = e.precision_at_k(j)
            precisions.append(p_at_k)
            r_at_k = e.recall_at_k(j)
            recalls.append(r_at_k)
            r_prec = e.r_precision(j)
            r_precisions.append(r_prec)
            avg_prec = e.avg_precision(j)
            avgs.append(avg_prec)
            out.write("{0:.3f}".format(p_at_k)+",")
            out.write("{0:.3f}".format(r_at_k)+",")
            out.write("{0:.3f}".format(r_prec)+",")
            out.write("{0:.3f}".format(avg_prec)+"\n")
        precisions = sum(precisions)/len(precisions)
        recalls = sum(recalls)/len(recalls)
        r_precisions = sum(r_precisions)/len(r_precisions)
        avgs = sum(avgs)/len(avgs)
        out.write(str(i) + "," + "mean," + "{0:.3f}".format(precisions) + ",")
        out.write("{0:.3f}".format(recalls) + ",")
        out.write("{0:.3f}".format(r_precisions) + ",")
        out.write("{0:.3f}".format(avgs)+"\n")
