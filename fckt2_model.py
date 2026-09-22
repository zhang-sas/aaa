"""FCKT 2.0 joint span and sentiment model.

The BERT and original extraction/classification parameter names are retained so
an old FCKT state dictionary can initialize the corresponding layers.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from bert.modeling import BertModel


def binary_entropy(logits):
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)


def span_iou(a, b):
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    intersection = max(0, right - left + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return intersection / union


class FCKT2(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        h = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_outputs = nn.Linear(h, 1)
        self.end_outputs = nn.Linear(h, 1)
        self.classifier = nn.Linear(h, 5)
        self.span_encoder = nn.Linear(h * 3, h)
        self.relation = nn.MultiheadAttention(h, num_heads=1, batch_first=True)
        self.relation_norm = nn.LayerNorm(h)
        self.prototypes = nn.Parameter(torch.empty(5, h))
        self.correction = nn.Sequential(nn.Linear(h + 2, h // 2), nn.ReLU(), nn.Linear(h // 2, 5))
        self.args = args
        self.apply(lambda m: self._init(m, config.initializer_range))
        nn.init.normal_(self.prototypes, std=config.initializer_range)
        # Create the optional head after common weights so paired runs start
        # with identical BERT and existing-head initializations for one seed.
        if getattr(args, 'use_context_evidence', False):
            self.evidence_dim = min(128, h)
            self.evidence_query = nn.Linear(h, self.evidence_dim, bias=False)
            self.evidence_key = nn.Linear(h, self.evidence_dim, bias=False)
            self.evidence_gate = nn.Linear(h * 2, 1)
            for module in (self.evidence_query, self.evidence_key, self.evidence_gate):
                self._init(module, config.initializer_range)
            nn.init.constant_(self.evidence_gate.bias, -2.0)

    @staticmethod
    def _init(module, std):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)

    def encode(self, input_ids, token_type_ids, attention_mask):
        layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        return self.dropout(layers[-1])

    def boundaries(self, hidden):
        return self.start_outputs(hidden).squeeze(-1), self.end_outputs(hidden).squeeze(-1)

    def spans(self, hidden, starts, ends, mask=None):
        """Inclusive, padded token spans; outputs [batch, aspects, hidden]."""
        b, length, h = hidden.shape
        starts = starts.clamp(0, length - 1)
        ends = ends.clamp(0, length - 1)
        at_start = hidden.gather(1, starts.unsqueeze(-1).expand(-1, -1, h))
        at_end = hidden.gather(1, ends.unsqueeze(-1).expand(-1, -1, h))
        positions = torch.arange(length, device=hidden.device)[None, None, :]
        within = (positions >= starts.unsqueeze(-1)) & (positions <= ends.unsqueeze(-1))
        pooled = torch.einsum('bml,blh->bmh', within.to(hidden.dtype), hidden)
        pooled = pooled / within.sum(-1).clamp_min(1).unsqueeze(-1)
        represented = torch.tanh(self.span_encoder(torch.cat((at_start, at_end, pooled), -1)))
        if mask is not None:
            represented = represented * mask.unsqueeze(-1)
        return represented

    def context_evidence(self, represented, hidden, attention_mask, aspect_mask=None):
        """Fuse sentence evidence selected separately for each aspect candidate."""
        query = self.evidence_query(represented)
        key = self.evidence_key(hidden)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.evidence_dim)
        valid_tokens = attention_mask.bool().clone()
        valid_tokens[:, 0] = False
        sep = attention_mask.long().sum(-1) - 1
        valid_tokens.scatter_(1, sep.clamp_min(0).unsqueeze(1), False)
        # Tiny synthetic inputs may contain only special tokens.
        no_words = ~valid_tokens.any(-1)
        if no_words.any():
            valid_tokens[no_words, 0] = True
        scores = scores.masked_fill(~valid_tokens[:, None, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, hidden)
        gate = torch.sigmoid(self.evidence_gate(torch.cat((represented, context), -1)))
        fused = represented + self.dropout(gate * context)
        if aspect_mask is not None:
            fused = fused * aspect_mask.unsqueeze(-1)
        return fused

    def classify(self, represented, mask=None, relational=True, hidden=None, attention_mask=None):
        if getattr(self.args, 'use_context_evidence', False):
            if hidden is None or attention_mask is None:
                raise ValueError('Context evidence requires hidden states and attention mask')
            represented = self.context_evidence(represented, hidden, attention_mask, mask)
        if relational and self.args.use_relation and represented.shape[1] > 1:
            key_mask = ~mask.bool() if mask is not None else None
            # MultiheadAttention cannot attend to a sentence with every position masked.
            if key_mask is not None:
                key_mask = key_mask.clone()
                key_mask[:, 0] = False
            related, _ = self.relation(represented, represented, represented,
                                       key_padding_mask=key_mask, need_weights=False)
            represented = self.relation_norm(represented + related)
        prototype_logits = F.normalize(represented, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        logits = self.classifier(represented) + prototype_logits
        return logits, represented

    def top_spans(self, start_logits, end_logits, attention_mask, k):
        """Select valid initial spans; no gold labels enter candidate generation."""
        b, length = start_logits.shape
        max_width = self.args.max_answer_length
        valid = attention_mask.bool().clone()
        valid[:, 0] = False
        sep = attention_mask.long().sum(-1) - 1
        valid.scatter_(1, sep.clamp_min(0).unsqueeze(1), False)
        scores = start_logits[:, :, None] + end_logits[:, None, :]
        idx = torch.arange(length, device=scores.device)
        span_valid = valid[:, :, None] & valid[:, None, :]
        span_valid = span_valid & (idx[None, :, None] <= idx[None, None, :])
        span_valid = span_valid & (idx[None, None, :] - idx[None, :, None] < max_width)
        flat = scores.masked_fill(~span_valid, -1e4).reshape(b, -1)
        values, positions = flat.topk(min(k, flat.shape[-1]), dim=-1)
        return positions // length, positions % length, values, values > -1e3

    def reliability(self, start_logits, end_logits, sentiment_logits, starts, ends, mask):
        hs = binary_entropy(start_logits.gather(1, starts))
        he = binary_entropy(end_logits.gather(1, ends))
        boundary_h = (hs + he) / 2
        probs = F.softmax(sentiment_logits, -1).clamp_min(1e-8)
        sentiment_h = -(probs * probs.log()).sum(-1) / math.log(5)
        score = torch.sigmoid(self.args.reliability_bias
                              - self.args.reliability_boundary * boundary_h
                              - self.args.reliability_sentiment * sentiment_h)
        return score.detach() * mask

    def candidate_spans(self, starts, ends, length):
        """Boundary shifts, expansions and truncations, including the anchor."""
        result = []
        for s, e in zip(starts, ends):
            options = {(s, e)}
            for ds, de in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, 1), (-1, 1), (1, -1)):
                ns, ne = s + ds, e + de
                if 1 <= ns <= ne < length - 1 and ne - ns + 1 <= self.args.max_answer_length:
                    options.add((ns, ne))
            result.append(sorted(options))
        return result

    def correction_logits(self, hidden, attention_mask, start_logits, end_logits, starts, ends):
        represented = self.spans(hidden, starts, ends)
        base_logits, represented = self.classify(
            represented, hidden=hidden, attention_mask=attention_mask)
        evidence = torch.stack((start_logits.gather(1, starts),
                                end_logits.gather(1, ends)), dim=-1)
        return base_logits + self.correction(torch.cat((represented, evidence), -1))

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None,
                end_positions=None, span_starts=None, span_ends=None, labels=None, label_masks=None):
        hidden = self.encode(input_ids, token_type_ids, attention_mask)
        start_logits, end_logits = self.boundaries(hidden)
        if labels is None:
            return hidden, start_logits, end_logits
        valid_tokens = attention_mask.bool().clone()
        valid_tokens[:, 0] = False
        sep = attention_mask.long().sum(-1) - 1
        valid_tokens.scatter_(1, sep.clamp_min(0).unsqueeze(1), False)
        start_loss = F.binary_cross_entropy_with_logits(start_logits[valid_tokens],
                                                         start_positions[valid_tokens].float())
        end_loss = F.binary_cross_entropy_with_logits(end_logits[valid_tokens],
                                                       end_positions[valid_tokens].float())
        valid = label_masks.bool()
        gold_reps = self.spans(hidden, span_starts, span_ends, valid.float())
        gold_logits, gold_reps = self.classify(
            gold_reps, valid, hidden=hidden, attention_mask=attention_mask)
        if valid.any():
            sentiment_loss = F.cross_entropy(gold_logits[valid], labels[valid])
        else:
            sentiment_loss = start_logits.sum() * 0
        loss = self.args.weight_start * start_loss + self.args.weight_end * end_loss
        loss = loss + self.args.weight_ac * sentiment_loss
        if valid.any() and self.args.weight_span > 0:
            normalized = F.normalize(hidden, dim=-1)
            anchors = normalized.gather(1, span_starts.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            similarities = torch.einsum('bmh,blh->bml', anchors, normalized) / 0.07
            similarities = similarities.masked_fill(~valid_tokens[:, None, :], -1e4)
            contrastive = F.cross_entropy(similarities[valid], span_ends[valid])
            loss = loss + self.args.weight_span * contrastive
        if self.args.use_reliability and valid.any():
            reliability = self.reliability(start_logits, end_logits, gold_logits,
                                           span_starts, span_ends, valid.float())
            pred_s, pred_e, _, pred_valid = self.top_spans(start_logits.detach(), end_logits.detach(),
                                                           attention_mask, self.args.n_best_size)
            pred_logits, _ = self.classify(
                self.spans(hidden, pred_s, pred_e), pred_valid,
                hidden=hidden, attention_mask=attention_mask)
            transfer_terms = []
            supervised_terms = []
            for bi in range(input_ids.shape[0]):
                choices = [(int(pred_s[bi, j]), int(pred_e[bi, j]))
                           for j in range(pred_s.shape[1]) if bool(pred_valid[bi, j])]
                for gi in valid[bi].nonzero(as_tuple=True)[0].tolist():
                    if not choices:
                        continue
                    gold = (int(span_starts[bi, gi]), int(span_ends[bi, gi]))
                    best = max(range(len(choices)), key=lambda j: span_iou(gold, choices[j]))
                    if span_iou(gold, choices[best]) < 0.5:
                        continue
                    r = reliability[bi, gi]
                    gold_ce = F.cross_entropy(gold_logits[bi, gi].unsqueeze(0),
                                              labels[bi, gi].unsqueeze(0))
                    predicted_ce = F.cross_entropy(pred_logits[bi, best].unsqueeze(0),
                                                   labels[bi, gi].unsqueeze(0))
                    supervised_terms.append(r * (predicted_ce - gold_ce))
                    teacher = F.softmax(gold_logits[bi, gi].detach() / self.args.kl_temperature, -1)
                    student = F.log_softmax(pred_logits[bi, best] / self.args.kl_temperature, -1)
                    transfer_terms.append(r * F.kl_div(student, teacher, reduction='sum'))
            if transfer_terms:
                loss = loss + self.args.weight_ac * torch.stack(supervised_terms).sum() / valid.sum()
                loss = loss + self.args.weight_kl * self.args.kl_temperature ** 2 * torch.stack(transfer_terms).mean()

        if self.args.use_relation and valid.any():
            relation_terms = []
            for bi in range(input_ids.shape[0]):
                embeddings = F.normalize(gold_reps[bi, valid[bi]], dim=-1)
                classes = labels[bi, valid[bi]]
                if embeddings.shape[0] < 2:
                    continue
                sim = embeddings @ embeddings.T
                same = classes[:, None] == classes[None, :]
                different = ~same
                eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
                positive = (1 - sim)[same & ~eye]
                negative = F.relu(sim[different] - self.args.relation_margin)
                terms = []
                if positive.numel():
                    terms.append(positive.mean())
                if negative.numel():
                    terms.append(negative.mean())
                if terms:
                    relation_terms.append(sum(terms))
            if relation_terms:
                loss = loss + self.args.weight_relation * torch.stack(relation_terms).mean()

        if self.args.use_correction:
            positive_terms = []
            negative_terms = []
            initial_s, initial_e, _, initial_valid = self.top_spans(
                start_logits.detach(), end_logits.detach(), attention_mask, self.args.n_best_size)
            for bi in range(input_ids.shape[0]):
                sentence_end = int(attention_mask[bi].sum())
                anchors = [(int(initial_s[bi, j]), int(initial_e[bi, j]))
                           for j in range(initial_s.shape[1]) if bool(initial_valid[bi, j])]
                gold = {(int(span_starts[bi, gi]), int(span_ends[bi, gi])): int(labels[bi, gi])
                        for gi in valid[bi].nonzero(as_tuple=True)[0].tolist()}
                options = set(gold)
                for anchor in anchors:
                    options.update(self.candidate_spans([anchor[0]], [anchor[1]], sentence_end)[0])
                if not options:
                    continue
                options = sorted(options)
                ss = torch.tensor([[s for s, _ in options]], device=hidden.device)
                ee = torch.tensor([[e for _, e in options]], device=hidden.device)
                logits = self.correction_logits(hidden[bi:bi + 1], attention_mask[bi:bi + 1],
                                                start_logits[bi:bi + 1],
                                                end_logits[bi:bi + 1], ss, ee)[0]
                targets = torch.tensor([gold.get(option, 0) for option in options], device=hidden.device)
                ce = F.cross_entropy(logits, targets, reduction='none')
                if (targets != 0).any():
                    positive_terms.append(ce[targets != 0].mean())
                if (targets == 0).any():
                    negative_terms.append(ce[targets == 0].mean())
            if positive_terms:
                loss = loss + self.args.weight_correction * torch.stack(positive_terms).mean()
            if negative_terms:
                loss = loss + self.args.weight_correction * torch.stack(negative_terms).mean()
        return loss

    @torch.no_grad()
    def predict(self, input_ids, token_type_ids, attention_mask):
        hidden, start, end = self.forward(input_ids, token_type_ids, attention_mask)
        starts, ends, boundary, valid = self.top_spans(start, end, attention_mask,
                                                        self.args.n_best_size)
        if not self.args.use_correction:
            logits, _ = self.classify(self.spans(hidden, starts, ends), valid,
                                      hidden=hidden, attention_mask=attention_mask)
        outputs = []
        for bi in range(input_ids.shape[0]):
            predictions = []
            sentence_end = int(attention_mask[bi].sum())
            if self.args.use_correction:
                options = set()
                for j in range(starts.shape[1]):
                    if bool(valid[bi, j]):
                        s, e = int(starts[bi, j]), int(ends[bi, j])
                        options.update(self.candidate_spans([s], [e], sentence_end)[0])
                if options:
                    options = sorted(options)
                    ss = torch.tensor([[s for s, _ in options]], device=hidden.device)
                    ee = torch.tensor([[e for _, e in options]], device=hidden.device)
                    scores = self.correction_logits(hidden[bi:bi + 1], attention_mask[bi:bi + 1],
                                                     start[bi:bi + 1], end[bi:bi + 1], ss, ee)[0]
                    margins, labels = scores[:, 1:].max(-1)
                    margins = margins - scores[:, 0]
                    for index, (s, e) in enumerate(options):
                        score = float(margins[index])
                        if score >= self.args.logit_threshold:
                            predictions.append((s, e, int(labels[index]) + 1, score))
            else:
                for j in range(starts.shape[1]):
                    if not bool(valid[bi, j]):
                        continue
                    s, e = int(starts[bi, j]), int(ends[bi, j])
                    label = int(logits[bi, j].argmax())
                    score = float(boundary[bi, j] + logits[bi, j, label])
                    if label != 0 and score >= self.args.logit_threshold:
                        predictions.append((s, e, label, score))
            # A span can be proposed by several anchors. Keep its highest score.
            unique = {}
            for item in predictions:
                key = item[:2]
                if key not in unique or item[3] > unique[key][3]:
                    unique[key] = item
            outputs.append(sorted(unique.values(), key=lambda x: x[3], reverse=True)[:self.args.n_best_size])
        return outputs
