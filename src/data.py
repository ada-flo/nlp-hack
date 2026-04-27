"""Dataset + DataLoader factory for the LSTM seq2seq."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
SEP_ID = 4


class Seq2SeqJsonl(Dataset):
    """Tokenizes encoder_input and decoder_target via SentencePiece.

    decoder_input is created here as [<SOS>] + tokens (without final <EOS>).
    decoder_target is tokens + [<EOS>].
    """

    def __init__(
        self,
        path: str | Path,
        sp: spm.SentencePieceProcessor,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        self.path = Path(path)
        self.sp = sp
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records: list[dict] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        src_ids = self.sp.encode_as_ids(r["encoder_input"])[: self.max_src_len]
        tgt_ids = self.sp.encode_as_ids(r["target_output"])[: self.max_tgt_len - 2]
        decoder_input = [SOS_ID] + tgt_ids
        decoder_target = tgt_ids + [EOS_ID]
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_target": torch.tensor(decoder_target, dtype=torch.long),
        }


def collate(batch):
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=PAD_ID)
    dec_in = pad_sequence([b["decoder_input"] for b in batch], batch_first=True, padding_value=PAD_ID)
    dec_tgt = pad_sequence([b["decoder_target"] for b in batch], batch_first=True, padding_value=PAD_ID)
    src_lens = (src != PAD_ID).sum(dim=1)
    return {
        "src": src,
        "src_lens": src_lens,
        "decoder_input": dec_in,
        "decoder_target": dec_tgt,
    }


def make_loaders(
    data_dir: str | Path = "data/processed",
    sp_path: str | Path = "data/processed/spm.model",
    batch_size: int = 32,
    num_workers: int = 2,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader, spm.SentencePieceProcessor]:
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))

    data_dir = Path(data_dir)
    train_ds = Seq2SeqJsonl(data_dir / "train.jsonl", sp, max_src_len, max_tgt_len)
    valid_ds = Seq2SeqJsonl(data_dir / "valid.jsonl", sp, max_src_len, max_tgt_len)
    test_ds = Seq2SeqJsonl(data_dir / "test.jsonl", sp, max_src_len, max_tgt_len)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, collate_fn=collate, pin_memory=True),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=collate, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=collate, pin_memory=True),
        sp,
    )


# ---------------------------------------------------------------------------
# XLM-R mode: src side uses XLM-R's tokenizer, tgt side stays on SentencePiece.
# ---------------------------------------------------------------------------


class Seq2SeqJsonlXLMR(Dataset):
    """Same dataset, but encoder_input is tokenized with XLM-R's tokenizer."""

    def __init__(self, path, sp, hf_tokenizer, max_src_len: int = 128,
                 max_tgt_len: int = 128):
        self.path = Path(path)
        self.sp = sp
        self.tok = hf_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records: list[dict] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        src = self.tok(
            r["encoder_input"], truncation=True, max_length=self.max_src_len,
            return_tensors="pt", add_special_tokens=True,
        )
        tgt_ids = self.sp.encode_as_ids(r["target_output"])[: self.max_tgt_len - 2]
        return {
            "src": src["input_ids"].squeeze(0),
            "src_attn": src["attention_mask"].squeeze(0),
            "decoder_input": torch.tensor([SOS_ID] + tgt_ids, dtype=torch.long),
            "decoder_target": torch.tensor(tgt_ids + [EOS_ID], dtype=torch.long),
        }


def collate_xlmr(batch, pad_id: int = 1):
    # XLM-R uses pad_id=1 in tokenizer.pad_token_id by default.
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=pad_id)
    src_attn = pad_sequence([b["src_attn"] for b in batch], batch_first=True, padding_value=0)
    dec_in = pad_sequence([b["decoder_input"] for b in batch], batch_first=True, padding_value=PAD_ID)
    dec_tgt = pad_sequence([b["decoder_target"] for b in batch], batch_first=True, padding_value=PAD_ID)
    return {
        "src": src,
        "attention_mask": src_attn,
        "decoder_input": dec_in,
        "decoder_target": dec_tgt,
    }


def make_loaders_xlmr(
    data_dir: str | Path = "data/processed",
    sp_path: str | Path = "data/processed/spm.model",
    hf_model: str = "xlm-roberta-base",
    batch_size: int = 16,
    num_workers: int = 2,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
):
    from transformers import AutoTokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))
    tok = AutoTokenizer.from_pretrained(hf_model)
    pad_id = tok.pad_token_id

    data_dir = Path(data_dir)
    train_ds = Seq2SeqJsonlXLMR(data_dir / "train.jsonl", sp, tok, max_src_len, max_tgt_len)
    valid_ds = Seq2SeqJsonlXLMR(data_dir / "valid.jsonl", sp, tok, max_src_len, max_tgt_len)
    test_ds = Seq2SeqJsonlXLMR(data_dir / "test.jsonl", sp, tok, max_src_len, max_tgt_len)

    def _collate(batch):
        return collate_xlmr(batch, pad_id=pad_id)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, collate_fn=_collate, pin_memory=True),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=_collate, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=_collate, pin_memory=True),
        sp,
        tok,
    )
