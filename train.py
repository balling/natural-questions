import random
import argparse
import numpy as np
import torch
import os
import pickle
import logging
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam

from setup import InputFeatures
import util

logger = util.get_logger('./tmp/', __name__)

def get_train_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default='./data/preprocessed/nq-train-01-features', type=str, help="Preprocessed train feature pickle files.")

    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                        "of training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, len(gpu_ids), args.fp16))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    model = BertForQuestionAnswering.from_pretrained(args.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))
    logger.info(model.config)

    if args.fp16:
        model.half()
    model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    
    with open(args.train_file, "rb") as reader:
        train_features = pickle.load(reader)
    num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                logger.info(loss)
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
    
    ckpt_dict = {
        'model_name': model.__class__.__name__,
        'model_state': model.cpu().state_dict(),
        'step': global_step
    }
    torch.save(ckpt_dict, './tmp/step_{}.pth.tar'.format(global_step))

if __name__ == "__main__":
    main()