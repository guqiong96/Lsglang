"""Aux hidden states captured for Eagle3/DFlash draft models."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStateList(list):
    """Retain aux snapshots independently of storage reused by later layers."""

    def capture(self, hidden: torch.Tensor, *, owned: bool = False) -> None:
        """Copy borrowed storage, or adopt a value whose ownership is transferred.

        Views and reusable communication buffers are borrowed even when they
        are different tensor objects from the source.
        """
        self.append(hidden if owned else hidden.clone())


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.

    ``for_batch`` writes into a decode graph runner's buffer when the batch
    carries one, so every graph size aliases it instead of pinning its own.
    """

    def __init__(
        self,
        num_captures: int,
        out: Optional[torch.Tensor] = None,
        rows: Optional[int] = None,
    ) -> None:
        self._num_captures = int(num_captures)
        self._buffer = out
        self._feature_size: Optional[int] = None
        self._idx = 0
        # Set only when a capture reaches this packer in row bands, i.e. while
        # the super-chunk wrapper loops the forward in inner chunks: the buffer
        # is CAP rows tall, which the first chunk's leading shape cannot tell
        # us. ``_col`` is the capture being filled and ``_filled`` the rows it
        # has taken so far, so a full band pass advances to the next column.
        self._rows = None if rows is None else int(rows)
        self._col = 0
        self._filled = 0

    @classmethod
    def for_batch(
        cls, forward_batch: ForwardBatch, num_captures: int
    ) -> AuxHiddenStatePacker:
        return cls(num_captures, out=forward_batch.aux_hidden_states_buffer)

    def append(
        self, hidden: torch.Tensor, *, rows: Optional[Tuple[int, int]] = None
    ) -> None:
        """Store one capture: the whole tensor, or one ``[lo, hi)`` band of it.

        The band form lets a model running inside the super-chunk inner loop
        hand a capture over piece by piece instead of materializing a CAP-shape
        copy of every captured layer; the destination is the same single column
        of the packed buffer either way. Bands must arrive grouped -- all of one
        capture before the next capture starts, which is what a layer loop does
        -- and a band that would overrun its column raises instead of quietly
        overwriting a neighbour's.
        """
        feature_size = int(hidden.shape[-1])
        if rows is None:
            if self._feature_size is None:
                self._feature_size = feature_size
                shape = (*hidden.shape[:-1], feature_size * self._num_captures)
                if self._buffer is None:
                    self._buffer = hidden.new_empty(shape)
                elif (
                    self._buffer.shape != shape or self._buffer.dtype != hidden.dtype
                ):
                    raise ValueError(
                        f"aux hidden output buffer is {tuple(self._buffer.shape)} "
                        f"{self._buffer.dtype}, captures need {shape} {hidden.dtype}"
                    )
            start = self._idx * self._feature_size
            self._buffer[..., start : start + self._feature_size].copy_(hidden)
            self._idx += 1
            return

        lo, hi = int(rows[0]), int(rows[1])
        if self._feature_size is None:
            self._feature_size = feature_size
            shape = (int(self._rows), feature_size * self._num_captures)
            if self._buffer is None:
                self._buffer = hidden.new_empty(shape)
            elif (
                tuple(self._buffer.shape) != shape or self._buffer.dtype != hidden.dtype
            ):
                raise ValueError(
                    f"aux hidden output buffer is {tuple(self._buffer.shape)} "
                    f"{self._buffer.dtype}, captures need {shape} {hidden.dtype}"
                )
        elif self._filled == self._rows:
            self._col += 1
            self._filled = 0
        if self._col >= self._num_captures or self._filled + (hi - lo) > self._rows:
            raise RuntimeError(
                f"aux capture bands out of order: column {self._col} of "
                f"{self._num_captures}, rows {self._filled} + {hi - lo} of "
                f"{self._rows}"
            )
        start = self._col * self._feature_size
        self._buffer[lo:hi, start : start + self._feature_size].copy_(hidden)
        self._filled += hi - lo

    def capture(
        self, hidden: torch.Tensor, *, owned: bool = False, rows: Optional[Tuple[int, int]] = None
    ) -> None:
        """Write directly to the final packed buffer, without an intermediate copy."""
        self.append(hidden, rows=rows)

    def _captures_done(self) -> int:
        """Captures fully stored, in either mode."""
        if self._rows is None:
            return self._idx
        return self._col + (1 if self._filled == self._rows else 0)

    def __len__(self) -> int:
        return self._captures_done()

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        done = self._captures_done()
        if self._feature_size is None or done != self._num_captures:
            raise RuntimeError(
                f"captured {done} of {self._num_captures} aux hidden states"
            )
        return self._buffer


# Both collectors accept borrowed values through capture; storage belongs to aux.
AuxHiddenStateAccumulator = Union[AuxHiddenStateList, AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return aux_hidden_states[0]
    return torch.cat(aux_hidden_states, dim=-1)
