import gzip
import glob
import os
import multiprocessing
import json
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pickle
from functools import partial
from util import get_logger
import argparse

logger = get_logger('.', 'setup')

def get_setup_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--is_eval", default=False, type=bool,
                        help="whether to downsample the candidates")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="convert text to lower case, True if using BERT uncased")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the processed examples and features will be written.")
    parser.add_argument("--input_path", default=None, type=str, required=True,
                        help="The natural questions dataset directory, e.g. ./data/natural_questions/v1.0/train/*.jsonl.gz")
    parser.add_argument("--n_threads", default=8, type=int, help="The number of threads for processing.")
    parser.add_argument('--max_seq_length', type=int, default=512, help="max feature length")
    parser.add_argument('--max_query_length', type=int, default=30, help="max query length")
    args = parser.parse_args()
    return args

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
                 original_start,
                 original_end,
                 has_answer=False,
                 start_position=-1,
                 end_position=-1,
                 yes_no=None):
        self.example_id = example_id
        self.candidate_id = candidate_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.original_start = original_start
        self.original_end = original_end
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
                 example_id,
                 candidate_id,
                 original_start,
                 original_end,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.example_id = example_id
        self.candidate_id = candidate_id
        self.original_start = original_start
        self.original_end = original_end
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
                    examples.append(NQExample(example_id, idx, question_text, candidate_tokens, candidate["start_token"], candidate["end_token"]))
            else:
                long_answer = long_answer_candidates.pop(candidate_index)
                long_answer_tokens = json_example["document_tokens"][long_answer["start_token"]:long_answer["end_token"]]
                if annotation["yes_no_answer"] != "NONE":
                    examples.append(NQExample(example_id,
                        candidate_index,
                        question_text,
                        long_answer_tokens,
                        long_answer["start_token"],
                        long_answer["end_token"],
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
                        long_answer["start_token"],
                        long_answer["end_token"],
                        has_answer=True,
                        start_position=short_answer["start_token"]-long_answer["start_token"],
                        end_position=short_answer["end_token"]-long_answer["start_token"]))
                else:
                    examples.append(NQExample(example_id, candidate_index, question_text, long_answer_tokens, long_answer["start_token"], long_answer["end_token"], has_answer=True))
                indexes = np.random.choice(range(num_candidates-1), min(9, num_candidates-1), replace=False)
                for idx in indexes:
                    candidate = long_answer_candidates[idx]
                    candidate_tokens = json_example["document_tokens"][candidate["start_token"]:candidate["end_token"]]
                    if idx >= candidate_index:
                        idx += 1
                    examples.append(NQExample(example_id, idx, question_text, candidate_tokens, candidate["start_token"], candidate["end_token"]))
    return examples

def read_nq_eval_examples(filename):
    """Read a natural question jsonl file into a list of NQExample."""
    examples = []
    with open(filename, "br") as fileobj:
        zip_file = gzip.GzipFile(fileobj=fileobj)
        for line in zip_file:
            json_example = json.loads(line)
            example_id = json_example["example_id"]
            question_text = json_example["question_text"]
            for idx, candidate in enumerate(json_example["long_answer_candidates"]):
                candidate_tokens = json_example["document_tokens"][candidate["start_token"]:candidate["end_token"]]
                examples.append(NQExample(example_id, idx, question_text, candidate_tokens, candidate["start_token"], candidate["end_token"]))
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
                example_id=example.example_id,
                candidate_id=example.candidate_id,
                original_start=example.original_start,
                original_end=example.original_end,
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
        logger.info("Saving {}...".format(message))
        with open(filename, "wb") as fh:
            pickle.dump(obj, fh)

def process_one_split(tokenizer, args, input_path):
    if args.is_eval:
        examples = read_nq_eval_examples(input_path)
    else:
        examples = read_nq_train_examples(input_path)
    prefix = os.path.basename(input_path).split('.jsonl')[0]
    file_name = os.path.join(args.output_dir, '{}.example'.format(prefix))
    save(file_name, examples, 'examples from {} to {}'.format(input_path, file_name))
    features = convert_examples_to_features(examples, tokenizer, args.max_seq_length, args.max_query_length)
    return features

def main():
    args = get_setup_args()
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    input_paths = glob.glob(args.input_path)
    pool = multiprocessing.Pool(args.n_threads)
    process = partial(process_one_split, tokenizer, args)
    try:
        features = pool.map(process, input_paths)
    finally:
        pool.close()
        pool.join()
    features = [feature for split in features for feature in split]
    dataset = 'eval' if args.is_eval else 'train'
    prefix = os.path.basename(input_paths[0]).split('.jsonl')[0]
    save(os.path.join(args.output_dir, '{}-{}-features-{}-{}'.format(prefix, dataset, args.max_seq_length, args.max_query_length)), features, '{} data features'.format(dataset))

if __name__ == '__main__':
    main()
