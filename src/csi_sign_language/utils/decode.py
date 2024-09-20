import torch
import ctcdecode
from torchtext.vocab import Vocab
from typing import Union, List, Literal, Any


class CTCBeamDecoder:
    def __init__(
        self,
        vocab: Vocab,
        blank_id=0,
        batch_first=False,
        log_probs_input=True,
        beam_width=10,
    ) -> None:
        self.vocab = vocab
        self.num_class = len(vocab)
        self.batch_first = batch_first
        self.log_probs_input = log_probs_input
        self.blank_id = blank_id
        self.beam_decoder = ctcdecode.CTCBeamDecoder(
            vocab.get_itos(),
            beam_width=beam_width,
            log_probs_input=self.log_probs_input,
            blank_id=blank_id,
            num_processes=10,
        )

    def __call__(self, emission: torch.Tensor, seq_length=None) -> Any:
        """
        :param probs: [n t c] if batch_first or [t n c]
        :param seq_length: [n] , defaults to None
        :return label a List[List[str], labels of decoded content
        :return beam_result [B, beam, timesteps]
        :return beam_scores [B, beam, timesteps]
        """

        if not self.batch_first:
            emission = emission.permute(1, 0, 2)  # force batch first
        emission = emission.cpu()
        if seq_length is not None:
            seq_length = seq_length.cpu()

        label = []
        beam_result, beam_scores, timesteps, out_seq_len = self.beam_decoder.decode(
            emission, seq_length
        )

        for batch_id in range(len(emission)):
            top_result = beam_result[batch_id][0][
                : out_seq_len[batch_id][0]
            ]  # [sequence_length]
            top_result = self.vocab.lookup_tokens(top_result.tolist())
            label.append(top_result)
        return label, beam_result, beam_scores, timesteps, out_seq_len
