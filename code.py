
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

    def precision_at_k(self, k=10):
        out = open("p@k.txt", 'a')
        counts = []
        for query in self.system_results:
            top_k_res = [values[0] for values in self.system_results[query][:k]]
            compare = [values[0] for values in self.relevant_docs[query]]
            count = 0
            for doc in top_k_res:
                if doc in compare:
                    count += 1
            counts.append(count/k)
            out.write(str(self.system_number) + "," + str(query) + "," + str(count/k) + "\n")
        mean = sum(counts)/k
        out.write(str(self.system_number) + "," + "mean" + "," + str(mean) + "\n")


if __name__ == '__main__':
    e = Eval(1, r"C:\Users\kenza\OneDrive\Documents\TTDS\qrels.csv", r"C:\Users\kenza\OneDrive\Documents\TTDS\system_results.csv")
    e.identify_system_results()
    e.precision_at_k()