import random
import argparse
import numpy as np
import torch
import os
import json
import math
import pickle
import logging
import collections
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from setup import InputFeatures
from model import BertForNQ
import util

logger = util.get_logger('./tmp/', __name__)

RawResult = collections.namedtuple("RawResult",
                                   ["example_id", "candidate_id", "start_logits", "end_logits", "type_logits"])


def write_predictions(all_features, all_results, output_prediction_file, n_best_size):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(dict)
    for feature in all_features:
        example_index_to_features[feature.example_id][feature.candidate_id] = feature

    example_index_to_results = collections.defaultdict(dict)
    for result in all_results:
        example_index_to_results[result.example_id][result.candidate_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["start_index", "end_index", "score"])

    predictions = []

    for (example_index, features) in example_index_to_features.items():
        long_score, candidate_id = max(
            (max(result.type_logits[1:]), c_id) for c_id, result in example_index_to_results[example_index].items()
        )
        candidate = features[candidate_id]
        long_start = candidate.original_start
        long_end = candidate.original_end
        result = example_index_to_results[example_index][candidate_id]
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        context_len = max(candidate.token_to_orig_map) - \
            min(candidate.token_to_orig_map)
        if result.type_logits[2] >= max(result.type_logits[2:]):  # YES
            predictions.append({
                'example_id': example_index,
                'long_answer': {
                    'start_token': long_start, 'end_token': long_end,
                    "start_byte": -1, "end_byte": -1
                },
                'long_answer_score': long_score,
                'short_answers': [{'start_token': -1, 'end_token': -1, "start_byte": -1, "end_byte": -1}],
                'short_answers_score': result.type_logits[2],
                'yes_no_answer': 'YES'
            })
            continue
        if result.type_logits[3] >= max(result.type_logits[2:]):  # NO
            predictions.append({
                'example_id': example_index,
                'long_answer': {
                    'start_token': long_start, 'end_token': long_end,
                    "start_byte": -1, "end_byte": -1
                },
                'long_answer_score': long_score,
                'short_answers': [{'start_token': -1, 'end_token': -1, "start_byte": -1, "end_byte": -1}],
                'short_answers_score': result.type_logits[3],
                'yes_no_answer': 'NO'
            })
            continue
        prelim_predictions = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index not in candidate.token_to_orig_map:
                    continue
                if end_index not in candidate.token_to_orig_map:
                    continue
                if end_index < start_index:
                    continue
                if end_index-start_index > 50:
                    continue
                # if end_index-start_index >= context_len:
                #     logger.warn('same length as long answer')
                #     continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        score=result.start_logits[start_index]+result.end_logits[end_index]-result.start_logits[0]-result.end_logits[0]))
        if len(prelim_predictions):
            pred = max(prelim_predictions, key=lambda x: x.score)
            short_start = long_start + \
                candidate.token_to_orig_map[pred.start_index]
            short_end = long_start + \
                candidate.token_to_orig_map[pred.end_index]+1
            predictions.append({
                'example_id': example_index,
                'long_answer': {
                    'start_token': long_start, 'end_token': long_end,
                    "start_byte": -1, "end_byte": -1
                },
                'long_answer_score': long_score,
                'short_answers': [{'start_token': short_start, 'end_token': short_end, "start_byte": -1, "end_byte": -1}],
                'short_answers_score': pred.score,
                'yes_no_answer': 'NONE'
            })
        else:
            predictions.append({
                'example_id': example_index,
                'long_answer': {
                    'start_token': long_start, 'end_token': long_end,
                    "start_byte": -1, "end_byte": -1
                },
                'long_answer_score': long_score,
                'short_answers': [{'start_token': -1, 'end_token': -1, "start_byte": -1, "end_byte": -1}],
                'short_answers_score': long_score,
                'yes_no_answer': 'NONE'
            })
    output = {"predictions": predictions}
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output, indent=4) + "\n")


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_train_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--predict_file", default='./data/preprocessed/nq-eval-features-512-30',
                        type=str, help="Preprocessed train feature pickle files.")
    parser.add_argument("--load_path", default=None, type=str,
                        help="Torch checkpoint to load from")
    parser.add_argument("--load_squad_path", default=None,
                        type=str, help="SQuAD pytorch model dir to load from")

    parser.add_argument("--n_best_size", default=6, type=int,
                        help="number of candidates to consider during evaluation")
    parser.add_argument("--train_batch_size", default=6,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=6,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                        "of training.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    return args


def main():
    args = get_train_args()
    device, gpu_ids = util.get_available_devices()

    # Set random seed
    logger.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.load_squad_path:
        output_model_file = os.path.join(args.load_squad_path, WEIGHTS_NAME)
        output_config_file = os.path.join(args.load_squad_path, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BertForNQ(config)
        model.load_state_dict(torch.load(output_model_file))
        # model.load_state_dict(torch.load(output_model_file, map_location='cpu')) # for local
    else:
        model = BertForNQ.from_pretrained(args.bert_model,
                                          cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)))
    logger.info(model.config)
    model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)

    if args.load_path:
        model, step = util.load_model(model, args.load_path, gpu_ids)

    with open(args.predict_file, "rb") as reader:
        eval_features = pickle.load(reader)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(
        set(f.example_id for f in eval_features)))
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, batch_type_logits = model(
                input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            type_logits = batch_type_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            all_results.append(RawResult(example_id=eval_feature.example_id,
                                         candidate_id=eval_feature.candidate_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         type_logits=type_logits))
    with open(os.path.join(args.output_dir, "all_results"), "wb") as file:
        pickle.dump(all_results, file)
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    write_predictions(eval_features, all_results,
                      output_prediction_file, args.n_best_size)


if __name__ == "__main__":
    main()
