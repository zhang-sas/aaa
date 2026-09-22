"""Train/evaluate FCKT 2.0 with the original SemEval files and BERT vocabulary."""
import argparse
import json
import logging
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bert.modeling import BertConfig
from bert.tokenization import FullTokenizer
from bert.fckt2_model import FCKT2
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, RawFinalResult
from absa.run_cls_span import eval_absa


LOG = logging.getLogger(__name__)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/absa')
    p.add_argument('--train_file', default='laptop14_train.txt')
    p.add_argument('--predict_file', default='laptop14_test.txt')
    p.add_argument('--bert_config_file', default='bert-large-uncased/bert_config.json')
    p.add_argument('--vocab_file', default='bert-large-uncased/vocab.txt')
    p.add_argument('--init_checkpoint', default='bert-large-uncased/pytorch_model.bin')
    p.add_argument('--output_dir', default='out/FCKT2')
    p.add_argument('--require_fresh_output', action='store_true',
                   help='Fail training if output_dir already contains a FCKT 2 checkpoint')
    p.add_argument('--do_train', action='store_true')
    p.add_argument('--do_predict', action='store_true')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--do_lower_case', action='store_true', default=True)
    p.add_argument('--verbose_logging', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_seq_length', type=int, default=96)
    p.add_argument('--max_answer_length', type=int, default=12)
    p.add_argument('--n_best_size', type=int, default=10)
    p.add_argument('--train_batch_size', type=int, default=2)
    p.add_argument('--predict_batch_size', type=int, default=4)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--num_train_epochs', type=float, default=5)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--head_learning_rate', type=float, default=5e-5)
    p.add_argument('--warmup_proportion', type=float, default=0.1)
    p.add_argument('--weight_start', type=float, default=1.0)
    p.add_argument('--weight_end', type=float, default=1.0)
    p.add_argument('--weight_span', type=float, default=1e-7)
    p.add_argument('--weight_ac', type=float, default=1.0)
    p.add_argument('--random_train', type=float, default=None)  # accepted, superseded by reliability
    p.add_argument('--logit_threshold', type=float, default=0.0)
    p.add_argument('--filter_type', default='f1')
    p.add_argument('--use_heuristics', action='store_true')
    p.add_argument('--use_nms', action='store_true')
    p.add_argument('--save_proportion', type=float, default=0.0)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--local_rank', type=int, default=-1)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--optimize_on_cpu', action='store_true')
    p.add_argument('--loss_scale', type=float, default=128.0)
    p.add_argument('--weight_kl', type=float, default=0.2)
    p.add_argument('--kl_temperature', type=float, default=2.0)
    p.add_argument('--reliability_bias', type=float, default=2.0)
    p.add_argument('--reliability_boundary', type=float, default=1.0)
    p.add_argument('--reliability_sentiment', type=float, default=1.0)
    p.add_argument('--weight_relation', type=float, default=0.1)
    p.add_argument('--relation_margin', type=float, default=0.2)
    p.add_argument('--weight_correction', type=float, default=0.1)
    p.add_argument('--correction_margin', type=float, default=0.5)
    p.add_argument('--use_reliability', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use_relation', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use_correction', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use_context_evidence', action='store_true',
                   help='Attend from each aspect candidate to sentence tokens for sentiment scoring')
    return p


def features_for(path, tokenizer, args):
    examples = convert_absa_data(read_absa_data(path), args.verbose_logging, keep_empty=True)
    features = convert_examples_to_features(examples, tokenizer, args.max_seq_length,
                                            args.verbose_logging, LOG)
    return examples, features


def optimizer_groups(model, args):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        base = name.startswith('bert.')
        decay = not (name.endswith('.bias') or 'LayerNorm' in name
                     or 'layer_norm' in name or name.endswith('norm.weight')
                     or name.endswith('norm.bias') or name == 'prototypes')
        key = (base, decay)
        groups.setdefault(key, []).append(param)
    grouped = [{'params': params, 'lr': args.learning_rate if base else args.head_learning_rate,
                'initial_lr': args.learning_rate if base else args.head_learning_rate,
                'weight_decay': 0.01 if decay else 0.0}
               for (base, decay), params in groups.items()]
    assert {id(p) for g in grouped for p in g['params']} == {
        id(p) for p in model.parameters() if p.requires_grad}
    return grouped


def tensor_data(features, training):
    columns = ['input_ids', 'input_mask', 'segment_ids']
    if training:
        columns += ['start_positions', 'end_positions', 'start_indexes',
                    'end_indexes', 'polarity_labels', 'label_masks']
    arrays = [torch.tensor([getattr(f, name) for f in features], dtype=torch.long)
              for name in columns]
    arrays.append(torch.arange(len(features)))
    return TensorDataset(*arrays)


def load_weights(model, path, device):
    loaded = torch.load(path, map_location=device, weights_only=False)
    weights = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded
    current = model.state_dict()
    compatible = {}
    for key, value in weights.items():
        clean = key.removeprefix('module.')
        if clean not in current and ('bert.' + clean) in current:
            clean = 'bert.' + clean
        if clean in current and current[clean].shape == value.shape:
            compatible[clean] = value
    missing = model.load_state_dict(compatible, strict=False).missing_keys
    LOG.info('Loaded %d tensors; %d newly initialized tensors: %s',
             len(compatible), len(missing), ', '.join(missing))
    return loaded if isinstance(loaded, dict) else {}


def evaluate(model, loader, examples, features, args, device):
    model.eval()
    results = []
    for batch in loader:
        ids, masks, segments, indexes = (x.to(device) for x in batch)
        predictions = model.predict(ids, segments, masks)
        for bi, index in enumerate(indexes.tolist()):
            feature = features[index]
            chosen = [x for x in predictions[bi]
                      if x[0] in feature.token_to_orig_map and x[1] in feature.token_to_orig_map]
            results.append(RawFinalResult(unique_id=feature.unique_id,
                                          start_indexes=[x[0] for x in chosen],
                                          end_indexes=[x[1] for x in chosen],
                                          cls_pred=[x[2] for x in chosen],
                                          span_masks=[1] * len(chosen)))
    metrics, predicted = eval_absa(examples, features, results,
                                   args.do_lower_case, args.verbose_logging, LOG)
    model.train()
    return metrics, predicted


def main():
    args = parser().parse_args()
    if not args.do_train and not args.do_predict and not args.eval_only:
        args.do_train = args.do_predict = True
    if args.eval_only:
        args.do_predict = True
    if args.local_rank != -1:
        raise ValueError('FCKT 2.0 runner currently supports one process; use a single GPU.')
    if args.fp16 or args.optimize_on_cpu:
        raise ValueError('Legacy fp16/CPU optimizer flags are unsupported; use standard PyTorch AMP externally.')
    if args.gradient_accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be >= 1')
    if args.num_train_epochs < 1 or not args.num_train_epochs.is_integer():
        raise ValueError('num_train_epochs must be a positive integer')
    if args.n_best_size < 1 or args.max_answer_length < 1:
        raise ValueError('n_best_size and max_answer_length must be positive')
    if not 0 <= args.warmup_proportion < 1:
        raise ValueError('warmup_proportion must be in [0, 1)')
    if not 0 <= args.save_proportion < 1:
        raise ValueError('save_proportion must be in [0, 1)')
    if args.do_train and os.path.abspath(os.path.join(args.data_dir, args.train_file)) == \
            os.path.abspath(os.path.join(args.data_dir, args.predict_file)):
        raise ValueError('Training and test files must differ')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    LOG.info('FCKT 2.0 device: %s; training file: %s', device, args.train_file)
    if device.type == 'cpu':
        LOG.warning('CUDA is unavailable; BERT Large training on CPU can be very slow.')
    os.makedirs(args.output_dir, exist_ok=True)
    for name in ('bert_config_file', 'vocab_file'):
        if not os.path.isfile(getattr(args, name)):
            raise FileNotFoundError('%s does not exist: %s' % (name, getattr(args, name)))
    config = BertConfig.from_json_file(args.bert_config_file)
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError('max_seq_length exceeds the BERT position limit')
    tokenizer = FullTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    save_path = os.path.join(args.output_dir, 'checkpoint_v2.pth.tar')
    legacy_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    if args.do_train and args.require_fresh_output:
        existing = [path for path in (save_path, legacy_path) if os.path.isfile(path)]
        if existing:
            raise FileExistsError('Fresh run requested but checkpoint already exists: %s' % existing[0])
    model = FCKT2(config, args).to(device)
    source = save_path if os.path.isfile(save_path) else (
        args.init_checkpoint if args.init_checkpoint and os.path.isfile(args.init_checkpoint)
        else (legacy_path if os.path.isfile(legacy_path) else None))
    if source is None and args.init_checkpoint:
        raise FileNotFoundError('Pretrained checkpoint does not exist: %s' % args.init_checkpoint)
    if args.eval_only and source is None:
        raise ValueError('Evaluation requires --init_checkpoint or a checkpoint in output_dir')
    checkpoint = load_weights(model, source, device) if source else {}
    if source == save_path and isinstance(checkpoint.get('config'), dict):
        saved_evidence = checkpoint['config'].get('use_context_evidence', False)
        if saved_evidence != args.use_context_evidence:
            raise ValueError('Checkpoint context-evidence setting differs from this run; '
                             'use the matching flag or a new output directory')
    LOG.info('Checkpoint source: %s; context evidence: %s', source, args.use_context_evidence)
    if args.do_train:
        train_examples, train_features = features_for(
            os.path.join(args.data_dir, args.train_file), tokenizer, args)
        train_generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(tensor_data(train_features, True), batch_size=args.train_batch_size,
                                  shuffle=True, generator=train_generator)
        LOG.info('Training on %d whole sentences with %d aspect labels', len(train_examples),
                 sum(len(x.term_texts) for x in train_examples))
    else:
        train_loader = None
    if args.do_train or args.do_predict:
        eval_examples, eval_features = features_for(os.path.join(args.data_dir, args.predict_file), tokenizer, args)
        eval_loader = DataLoader(tensor_data(eval_features, False), batch_size=args.predict_batch_size)
    else:
        eval_loader = None
    if args.do_train:
        optimizer = torch.optim.AdamW(optimizer_groups(model, args))
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_steps = max(1, int(args.num_train_epochs) * steps_per_epoch)
        warmup_steps = int(total_steps * args.warmup_proportion)
        global_step = 0
        best = -1.0
        model.train()
        optimizer.zero_grad()
        for epoch in range(int(args.num_train_epochs)):
            for bi, batch in enumerate(train_loader):
                ids, masks, segments, start, end, spans_s, spans_e, labels, label_masks, _ = (
                    x.to(device) for x in batch)
                loss = model(ids, segments, masks, start, end, spans_s, spans_e, labels, label_masks)
                group_start = (bi // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                group_size = min(args.gradient_accumulation_steps, len(train_loader) - group_start)
                (loss / group_size).backward()
                if (bi + 1) % args.gradient_accumulation_steps == 0 or bi + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    global_step += 1
                    if warmup_steps and global_step <= warmup_steps:
                        factor = global_step / warmup_steps
                    else:
                        factor = max(0.0, (total_steps - global_step + 1) /
                                     max(1, total_steps - warmup_steps))
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * factor
                    optimizer.step()
                    optimizer.zero_grad()
                if args.debug:
                    break
            metrics, _ = evaluate(model, eval_loader, eval_examples, eval_features, args, device)
            LOG.info('epoch=%d loss=%.4f joint_F1=%.4f', epoch + 1, float(loss), metrics['f1_all'])
            if metrics['f1_all'] >= best:
                best = metrics['f1_all']
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': global_step, 'best_f1': best, 'fckt_version': 2,
                            'config': vars(args)}, save_path)
            if args.debug:
                break
        load_weights(model, save_path, device)
    if args.do_predict:
        metrics, predictions = evaluate(model, eval_loader, eval_examples, eval_features, args, device)
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        LOG.info('P_all=%.4f R_all=%.4f F1_all=%.4f',
                 metrics['p_all'], metrics['r_all'], metrics['f1_all'])


if __name__ == '__main__':
    main()
