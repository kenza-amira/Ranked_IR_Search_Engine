
class Eval:
    def __init__(self, system_number, qrels_file):
        self.system_number = system_number
        self.relevant_docs = self.read_file(qrels_file)

    def read_file(cls, qrels_file):
        parsing = dict()
        with open(qrels_file, 'r', encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                line = line.strip().split(",")
                print(line)
                query_no = int(line[0])
                doc_no = int(line[1])
                relevance = int(line[2])
                if query_no in parsing:
                    parsing[query_no].append((doc_no, relevance))
                else:
                    parsing[query_no] = [(doc_no, relevance)]
        return parsing


if __name__ == '__main__':
    e = Eval(1, r"C:\Users\kenza\OneDrive\Documents\TTDS\qrels.csv")