import math

import torch

import torch.nn.functional as F

from tell.utils import strip_pad

from .base import Criterion

from tell.modules import GehringLinear


@Criterion.register('mm_adaptive_loss')
class MMAdaptiveLoss(Criterion):
    """Create the loss for the adaptive softmax approximation.

    This is an implementation of the loss function accompanying the adaptive
    softmax approximation for graphical processing units (GPU), described in
    the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, padding_idx=1, input_dim = 1024, method = "softmax"):
        super().__init__()
        self.padding_idx = padding_idx
        # normalize gradients by the number of sentences in a batch
        # (default is to normalize by number of tokens)
        self.proj_fake = GehringLinear(input_dim, 1)
        self.proj_mm = GehringLinear(input_dim * 2, 1)
        self.sentence_avg = False
        self.method = method
        
    def forward(self, adaptive_softmax, net_output, decoder_target, caption_ids, fake_captions_ids, reduction='sum'):
        if self.method == "sigmoid":
            return self.forward_sigmoid(adaptive_softmax, net_output, decoder_target, caption_ids, fake_captions_ids, reduction)
        elif self.method == "softmax":
            return self.forward_softmax(adaptive_softmax, net_output, decoder_target, caption_ids, fake_captions_ids, reduction)

    def forward_sigmoid(self, adaptive_softmax, net_output, decoder_target, caption_ids, article_indices, reduction='sum'):
        """Compute the loss for the given sample.

        Reduction can be 'sum' or None

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        orig_target = decoder_target
        # orig_target.shape == [batch_size, seq_len]

        orig_target = orig_target.reshape(-1)
        # orig_target.shape == [batch_size * seq_len]

        batch_size = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)
        # len(target) == len(logits) == n_clusters
        # logits[i].shape == [batch_size * seq_len, cluster_size]
        # target[i].shape == [batch_size * seq_len]

        loss = net_output[0].new(
            1 if reduction == 'sum' else batch_size).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert (target[i].min() >= 0 and target[i].max()
                        <= logits[i].size(1))
                loss += F.cross_entropy(logits[i], target[i], ignore_index=self.padding_idx,
                                        reduction=reduction)
        
        ##################################################################
        eos = 2
        eos_unmask = caption_ids == eos
        
        X = net_output[0]
        btz, _, dim = X.shape
        X = X[eos_unmask]
        
        mm_logits = []
        for i in range(btz):
            x = X.narrow(0, i, 1)
            x = x.expand(btz, dim)
            mm_logit = torch.cat([X, x], dim=1)
            mm_logit = self.proj_mm(mm_logit)
            mm_logits.append(mm_logit)
        mm_logits = torch.cat(mm_logits, 1)
        mm_unmask = torch.triu(mm_logits.new(torch.ones(mm_logits.shape).half().cuda()).bool(),diagonal=1)
        mm_logits = mm_logits[mm_unmask]
        mm_logits = torch.reshape(mm_logits, [-1])
        #print("mm_unmask", mm_unmask)
        
        mm_targets = []
        for i in range(btz):
            article_index = article_indices.narrow(0, i, 1)
            mm_target = article_indices == article_index
            mm_targets.append(mm_target)
        mm_targets = torch.stack(mm_targets, 0)
        mm_targets = mm_targets[mm_unmask]
        mm_targets = torch.reshape(mm_targets, [-1])
        mm_targets = mm_targets.half()
        #print(mm_targets)
        mm_loss = F.binary_cross_entropy_with_logits(mm_logits, mm_targets, reduction=reduction)
        
        ##################################################################

        orig = strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = decoder_target.size(
            0) if self.sentence_avg else ntokens
        mm_size = mm_targets.size(0)
        logging_output = {
            'loss': loss.data.item() if reduction == 'sum' else loss.data,
            'ntokens': ntokens,
            'nsentences': btz,
            'sample_size': sample_size,
            'mm_size': mm_size
        }
        
        return loss, sample_size, mm_loss, mm_size

    def forward_softmax(self, adaptive_softmax, net_output, decoder_target, caption_ids, fake_captions_ids, reduction='sum'):
        """Compute the loss for the given sample.

        Reduction can be 'sum' or None

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        orig_target = decoder_target
        # orig_target.shape == [batch_size, seq_len]

        orig_target = orig_target.reshape(-1)
        # orig_target.shape == [batch_size * seq_len]

        batch_size = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)
        # len(target) == len(logits) == n_clusters
        # logits[i].shape == [batch_size * seq_len, cluster_size]
        # target[i].shape == [batch_size * seq_len]

        loss = net_output[0].new(
            1 if reduction == 'sum' else batch_size).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert (target[i].min() >= 0 and target[i].max()
                        <= logits[i].size(1))
                loss += F.cross_entropy(logits[i], target[i], ignore_index=self.padding_idx,
                                        reduction=reduction)
                
        ##################################################################
        eos = 2
        eos_unmask = caption_ids == eos
        fake_eos_unmask = fake_captions_ids == eos
        
        X = net_output[0][eos_unmask]
        X = X.unsqueeze(1)
        
        Y = net_output[-1]
        btz, fakes, _, _ = Y.shape
        Y = Y[fake_eos_unmask]
        Y = Y.view(btz, fakes, -1)
        
        fake_logits = torch.cat([X, Y], dim=1)
        #fake_logits = torch.stack([net_output[0][eos_unmask], net_output[-1][fake_eos_unmask]], dim=1)
        fake_logits = self.proj_fake(fake_logits).squeeze(2) 
        #print(fake_logits)
        fake_target = fake_logits.new(fake_logits.size(0)).long().zero_()
        #fake_target[:, 0] = 1
        #fake_target = fake_target.transpose(0, 1)
        fake_loss = F.cross_entropy(fake_logits, fake_target, reduction=reduction)
        ##################################################################

        orig = strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = decoder_target.size(
            0) if self.sentence_avg else ntokens
        logging_output = {
            'loss': loss.data.item() if reduction == 'sum' else loss.data,
            'ntokens': ntokens,
            'nsentences': batch_size,
            'sample_size': sample_size,
        }
        
        return loss, sample_size, fake_loss, batch_size

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'nll_loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
