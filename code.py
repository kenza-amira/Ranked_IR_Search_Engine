import numpy as np


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

    def compute_dg(self, scores):
        dcg = []
        for idx, score in scores:
            res = score/np.log2(idx)
            dcg.append(res)
        return sum(dcg)

    def dcg_at_k(self, query, k):
        retrieved = [values[0] for values in self.system_results[query][:k]]
        scores = []
        dcg = 0
        for idx in range(len(retrieved)):
            for i, v in self.relevant_docs[query]:
                if i == retrieved[idx]:
                    scores.append((idx+1, v))
                    break
        if scores and scores[0][0] == 1:
            dcg += scores[0][1]
            dcg += self.compute_dg(scores[1:])
        else:
            dcg = self.compute_dg(scores)
        return dcg

    def idcg_at_k(self, query, k):
        scores = [values[1] for values in self.relevant_docs[query][:k]]
        scores = [(idx+1, value) for idx, value in enumerate(scores)]
        return scores[0][1] + self.compute_dg(scores[1:])

    def ndcg_at_k(self, query, k):
        return self.dcg_at_k(query, k=k)/self.idcg_at_k(query, k=k)


if __name__ == '__main__':
    out = open("ir_eval.csv", "w")
    out.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n")
    for i in range(1, 7):
        precisions = []
        recalls = []
        r_precisions = []
        avgs = []
        dcgs_10 = []
        dcgs_20 = []
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
            dcg_10 = e.ndcg_at_k(j, 10)
            dcgs_10.append(dcg_10)
            dcg_20 = e.ndcg_at_k(j, 20)
            dcgs_20.append(dcg_20)
            out.write("{0:.3f}".format(p_at_k)+",")
            out.write("{0:.3f}".format(r_at_k)+",")
            out.write("{0:.3f}".format(r_prec)+",")
            out.write("{0:.3f}".format(avg_prec)+",")
            out.write("{0:.3f}".format(dcg_10)+",")
            out.write("{0:.3f}".format(dcg_20)+"\n")
        precisions = sum(precisions)/len(precisions)
        recalls = sum(recalls)/len(recalls)
        r_precisions = sum(r_precisions)/len(r_precisions)
        avgs = sum(avgs)/len(avgs)
        dcgs_10 = sum(dcgs_10)/len(dcgs_10)
        dcgs_20 = sum(dcgs_20)/len(dcgs_20)
        out.write(str(i) + "," + "mean," + "{0:.3f}".format(precisions) + ",")
        out.write("{0:.3f}".format(recalls) + ",")
        out.write("{0:.3f}".format(r_precisions) + ",")
        out.write("{0:.3f}".format(avgs) + ",")
        out.write("{0:.3f}".format(dcgs_10) + ",")
        out.write("{0:.3f}".format(dcgs_20) + "\n")
