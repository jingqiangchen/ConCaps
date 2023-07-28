from typing import Dict

from allennlp.data.fields import IndexField
from overrides import overrides


class IndexField(IndexField):

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            'index': self.sequence_index
        }

