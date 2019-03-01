import base64
import gzip
import json
import numpy as np
from tqdm import tqdm

def process_train_file(filename="./data/v1.0_sample_nq-train-sample.jsonl.gz"):
    examples = []
    with open(filename, "br") as fileobj:
        zip_file = gzip.GzipFile(fileobj=fileobj)
        for line in zip_file:
            json_example = json.loads(line)
            long_answer_candidates = json_example["long_answer_candidates"]
            candidate_strs = []
            question_str = " ".join(json_example["question_tokens"])
            candidate_index = json_example["annotations"][0]['long_answer']['candidate_index']
            if candidate_index == -1: # long answer does not exist
                candidates = np.random.choice(long_answer_candidates, min(10, len(long_answer_candidates)), replace=False)
                long_answer_indices=[]
            else:
                candidates = [long_answer_candidates.pop(candidate_index)]
                candidates.extend(np.random.choice(long_answer_candidates, min(9, len(long_answer_candidates)), replace=False))
                long_answer_indices=[0]
            for candidate in candidates:
                candidate_tokens = json_example["document_tokens"][
                    candidate["start_token"]:candidate["end_token"]]
                candidate_tokens = [t["token"] for t in candidate_tokens]
                candidate_str = " ".join(candidate_tokens)
                candidate_strs.append(candidate_str)
            examples.append({
                "context": candidate_strs,
                "question": [question_str],
                "long_answer_indices": long_answer_indices
            })
        print("{} questions in total, {} candidates".format(len(examples), sum(len(e['context']) for e in examples)))
    return examples

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def main():
    examples = process_train_file('./data/natural_questions/v1.0/train/nq-train-00.jsonl.gz')
    save('./data/preprocessed/nq-train-00.json', examples, 'train data')


if __name__ == '__main__':
    main()
