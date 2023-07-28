import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque, Optional, Iterator

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values, is_lazy, ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
from django.utils.timezone import override

import math

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def split_and_sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0,
                    batch_size:int = 10) -> List[Instance]:

    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths_with_noise = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key] for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    
    instances_with_lengths.sort(key=lambda x: (x[0]))
    old_list = deque([item[-1] for item in instances_with_lengths])
    new_list = []
    ins_num = len(old_list)
    half_batch_size = batch_size
    for _ in range(ins_num // batch_size):
        for _ in range(half_batch_size):
            new_list.append(old_list.pop())
    new_list.extend(old_list)
    
    return new_list


@DataIterator.register("mm_iterator2")
class MMIterator2(DataIterator):

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False) -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._skip_smaller_batches = skip_smaller_batches

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):
            instance_list = split_and_sort_by_padding(instance_list,
                                                      self._sorting_keys,
                                                      self.vocab,
                                                      self._padding_noise,
                                                      self._batch_size)

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size): 
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    if self._skip_smaller_batches and len(possibly_smaller_batches) < self._batch_size:
                        continue
                    batch = Batch(possibly_smaller_batches)
                    batches.append(batch)
            if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                batches.append(Batch(excess))
            
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)
            
            yield from batches


    
