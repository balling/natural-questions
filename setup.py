import gzip
import json
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pickle

class NQExample(object):
    """
    A single training example for the Natural Question dataset.
    The examples can either be:
    1. No answer - has_answer=False, start and end position are -1, yes_no=None
    2. Only long answer - has_answer=True, start and end position are -1, yes_no=None
    3. Yes/No short answer - has_answer=True, start and end position are -1, yes_no=True/False
    4. Short answer - has_answer=True, non -1 start and end position, yes_no=None
    """

    def __init__(self,
                 example_id,
                 candidate_id,
                 question_text,
                 doc_tokens,
                 has_answer=False,
                 start_position=-1,
                 end_position=-1,
                 yes_no=None):
        self.example_id = example_id
        self.candidate_id = candidate_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.has_answer = has_answer
        self.start_position = start_position
        self.end_position = end_position
        self.yes_no = yes_no
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ""
        s += "id: %s-%s" % (self.example_id, self.candidate_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s...]" % (" ".join(t["token"] for t in self.doc_tokens[:5]))
        if not self.has_answer:
            s += ", no answer"
        else:
            if self.yes_no is not None:
                s += ", yes/no answer %s" % self.yes_no
            elif self.start_position>=0:
                s += ", short answer %d-%d" % (self.start_position, self.end_position)
            else:
                s += ", long answer"
        return s

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 id,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.id = id
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

def read_nq_train_examples(filename):
    """Read a natural question jsonl file into a list of NQExample."""
    examples = []
    with open(filename, "br") as fileobj:
        zip_file = gzip.GzipFile(fileobj=fileobj)
        for line in zip_file:
            json_example = json.loads(line)
            example_id = json_example["example_id"]
            question_text = json_example["question_text"]
            annotation = json_example["annotations"][0]
            candidate_index = annotation['long_answer']['candidate_index']
            long_answer_candidates = json_example["long_answer_candidates"]
            num_candidates = len(long_answer_candidates)
            if candidate_index == -1: # long answer does not exist
                indexes = np.random.choice(range(num_candidates), min(10, num_candidates), replace=False)
                for idx in indexes:
                    candidate = long_answer_candidates[idx]
                    candidate_tokens = json_example["document_tokens"][candidate["start_token"]:candidate["end_token"]]
                    examples.append(NQExample(example_id, idx, question_text, candidate_tokens))
            else:
                long_answer = long_answer_candidates.pop(candidate_index)
                long_answer_tokens = json_example["document_tokens"][long_answer["start_token"]:long_answer["end_token"]]
                if annotation["yes_no_answer"] != "NONE":
                    examples.append(NQExample(example_id,
                        candidate_index,
                        question_text,
                        long_answer_tokens,
                        has_answer=True,
                        start_position=-1,
                        end_position=-1,
                        yes_no=annotation["yes_no_answer"]))
                elif len(annotation["short_answers"])>0:
                    short_answer = annotation["short_answers"][0]
                    examples.append(NQExample(example_id,
                        candidate_index,
                        question_text,
                        long_answer_tokens,
                        has_answer=True,
                        start_position=short_answer["start_token"]-long_answer["start_token"],
                        end_position=short_answer["end_token"]-long_answer["start_token"]))
                else:
                    examples.append(NQExample(example_id, candidate_index, question_text, candidate_tokens, has_answer=True))
                indexes = np.random.choice(range(num_candidates-1), min(9, num_candidates-1), replace=False)
                for idx in indexes:
                    candidate = long_answer_candidates[idx]
                    candidate_tokens = json_example["document_tokens"][candidate["start_token"]:candidate["end_token"]]
                    if idx >= candidate_index:
                        idx += 1
                    examples.append(NQExample(example_id, idx, question_text, candidate_tokens))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length=30):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example in examples:
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if i>0 and token["html_token"]:
                continue # only keep the first html tag
            if token["token"].startswith("http"):
                continue # ignore URLs
            sub_tokens = tokenizer.tokenize(token["token"])
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if example.has_answer:
            if example.yes_no == "YES":
                # left [SEP]
                tok_start_position = -1
                tok_end_position = -1
            elif example.yes_no == "NO":
                # right [SEP]
                tok_start_position = len(all_doc_tokens)
                tok_end_position = len(all_doc_tokens)
            elif example.start_position>-1:
                # from left to right [SEP]
                tok_start_position = -1
                tok_end_position = len(all_doc_tokens)
            else:
                # actual short answer span
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # for simplicity truncate longer context for now
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        doc_offset = len(query_tokens) + 2
        token_to_orig_map = {(k+doc_offset):v for k, v in enumerate(tok_to_orig_index[:max_tokens_for_doc])}
        doc_tokens = all_doc_tokens[:max_tokens_for_doc]
        tokens.extend(doc_tokens)
        segment_ids.extend([1]*len(doc_tokens))
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        start_position = None
        end_position = None
        if tok_start_position is None:
            start_position = end_position = 0
        else:
            start_position = doc_offset + tok_start_position
            end_position = doc_offset + tok_end_position
            if start_position >= max_seq_length:
                # short answer is completely chopped off, switch to long answer
                start_position = doc_offset-1
                end_position = max_seq_length-1
            elif end_position >= max_seq_length:
                # short answer is partly chopped off, clip the end position
                end_position = max_seq_length-1
        features.append(
            InputFeatures(
                id= "%s-%s" % (example.example_id, example.candidate_id),
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position))
    return features

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "wb") as fh:
            pickle.dump(obj, fh)

def main():
    examples = read_nq_train_examples("./data/natural_questions/v1.0/train/nq-train-01.jsonl.gz")
    # examples = read_nq_train_examples("./data/v1.0_sample_nq-train-sample.jsonl.gz")
    save('./data/preprocessed/nq-train-01-examples', examples, 'train data examples')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    features = convert_examples_to_features(examples, tokenizer, 512)
    save('./data/preprocessed/nq-train-01-features', features, 'train data features')

if __name__ == '__main__':
    main()
