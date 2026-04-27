"""LSTM seq2seq with Bahdanau attention.

The encoder is selectable via `encoder_type`:
  - "bilstm"  → from-scratch BiLSTM (Phase 2 baseline).
  - "xlmr"    → frozen xlm-roberta-base providing encoder hidden states
                (Phase 3 hybrid). Decoder is always LSTM-based.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import PAD_ID, SOS_ID, EOS_ID


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3,
                 pad_id: int = PAD_ID, embedding: Optional[nn.Embedding] = None):
        super().__init__()
        if embedding is None:
            embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embedding = embedding
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project bidirectional state down to a single decoder-init vector.
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2  # encoder hidden states for attention

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # h: (num_layers*2, B, H) → take last layer's two directions, concat.
        last_h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2H)
        last_c = torch.cat([c[-2], c[-1]], dim=-1)
        dec_h0 = torch.tanh(self.bridge_h(last_h))  # (B, H)
        dec_c0 = torch.tanh(self.bridge_c(last_c))
        return outputs, (dec_h0, dec_c0)


class XLMREncoder(nn.Module):
    """Frozen xlm-roberta-base feeding into a Bahdanau-attention LSTM decoder.

    Note: this encoder uses XLM-R's own tokenizer, so the input ids you pass
    must come from `transformers.AutoTokenizer.from_pretrained('xlm-roberta-base')`,
    NOT the SentencePiece tokenizer used for the decoder side.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3,
                 model_name: str = "xlm-roberta-base"):
        super().__init__()
        from transformers import AutoModel
        self.xlmr = AutoModel.from_pretrained(model_name)
        for p in self.xlmr.parameters():
            p.requires_grad = False
        self.xlmr.eval()
        self.proj_out = nn.Linear(self.xlmr.config.hidden_size, hidden_dim * 2)
        self.bridge_h = nn.Linear(self.xlmr.config.hidden_size, hidden_dim)
        self.bridge_c = nn.Linear(self.xlmr.config.hidden_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, src, attention_mask):
        with torch.no_grad():
            out = self.xlmr(input_ids=src, attention_mask=attention_mask)
        seq = out.last_hidden_state  # (B, T, H_xlmr)
        encoder_outputs = self.proj_out(self.dropout(seq))  # (B, T, 2H)
        # Pool with attention_mask for decoder init.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        dec_h0 = torch.tanh(self.bridge_h(pooled))
        dec_c0 = torch.tanh(self.bridge_c(pooled))
        return encoder_outputs, (dec_h0, dec_c0)


# ---------------------------------------------------------------------------
# Bahdanau attention
# ---------------------------------------------------------------------------


class BahdanauAttention(nn.Module):
    def __init__(self, dec_dim: int, enc_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_state, enc_outputs, src_mask):
        # dec_state: (B, H), enc_outputs: (B, T, 2H), src_mask: (B, T) bool, True=keep
        scores = self.v(torch.tanh(
            self.W_dec(dec_state).unsqueeze(1) + self.W_enc(enc_outputs)
        )).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, 2H)
        return context, weights


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 enc_dim: int, num_layers: int = 1, dropout: float = 0.3,
                 pad_id: int = PAD_ID, tie_weights: bool = True,
                 embedding: Optional[nn.Embedding] = None):
        super().__init__()
        if embedding is None:
            embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embedding = embedding
        self.attn = BahdanauAttention(hidden_dim, enc_dim, attn_dim=256)
        self.lstm = nn.LSTM(
            embed_dim + enc_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim + enc_dim + embed_dim, embed_dim)
        self.out_logits = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights and embed_dim == self.embedding.embedding_dim:
            self.out_logits.weight = self.embedding.weight
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, prev_token, prev_state, enc_outputs, src_mask):
        # prev_token: (B,), prev_state: (h, c) each (1, B, H)
        emb = self.dropout(self.embedding(prev_token)).unsqueeze(1)  # (B, 1, E)
        h_top = prev_state[0][-1]  # (B, H)
        context, attn_weights = self.attn(h_top, enc_outputs, src_mask)  # (B, 2H)
        lstm_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, new_state = self.lstm(lstm_in, prev_state)
        # readout: combine RNN output, context, prev embedding.
        readout = self.out_proj(torch.cat([
            out.squeeze(1), context, emb.squeeze(1)
        ], dim=-1))
        logits = self.out_logits(self.dropout(torch.tanh(readout)))
        return logits, new_state, attn_weights


# ---------------------------------------------------------------------------
# Full seq2seq
# ---------------------------------------------------------------------------


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512,
                 enc_layers: int = 2, dec_layers: int = 1, dropout: float = 0.3,
                 encoder_type: str = "bilstm", tie_weights: bool = True):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(
                vocab_size, embed_dim, hidden_dim, num_layers=enc_layers,
                dropout=dropout,
            )
        elif encoder_type == "xlmr":
            self.encoder = XLMREncoder(hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"unknown encoder_type {encoder_type!r}")

        # Share decoder embedding with encoder embedding only when both
        # use the same SP vocab (i.e. bilstm encoder).
        share_emb = encoder_type == "bilstm"
        decoder_embedding = self.encoder.embedding if share_emb else None
        self.decoder = LSTMDecoder(
            vocab_size, embed_dim, hidden_dim, enc_dim=self.encoder.out_dim,
            num_layers=dec_layers, dropout=dropout,
            tie_weights=tie_weights, embedding=decoder_embedding,
        )

    def encode(self, src, src_lens=None, attention_mask=None):
        if self.encoder_type == "bilstm":
            return self.encoder(src, src_lens)
        return self.encoder(src, attention_mask)

    def forward(self, src, decoder_input, src_lens=None, attention_mask=None):
        enc_outputs, (h0, c0) = self.encode(src, src_lens, attention_mask)
        if attention_mask is not None:
            src_mask = attention_mask.bool()
        else:
            src_mask = src != PAD_ID

        # Stack decoder init across layers.
        n_layers = self.decoder.lstm.num_layers
        h = h0.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        c = c0.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        state = (h, c)

        all_logits = []
        for t in range(decoder_input.size(1)):
            tok = decoder_input[:, t]
            logits, state, _ = self.decoder.forward_step(tok, state, enc_outputs, src_mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)  # (B, T, V)

    @torch.no_grad()
    def greedy_decode(self, src, src_lens=None, attention_mask=None,
                      max_len: int = 64, sos_id: int = SOS_ID, eos_id: int = EOS_ID):
        self.eval()
        enc_outputs, (h0, c0) = self.encode(src, src_lens, attention_mask)
        if attention_mask is not None:
            src_mask = attention_mask.bool()
        else:
            src_mask = src != PAD_ID

        n_layers = self.decoder.lstm.num_layers
        h = h0.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        c = c0.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        state = (h, c)

        B = src.size(0)
        cur = torch.full((B,), sos_id, device=src.device, dtype=torch.long)
        outputs = []
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, state, _ = self.decoder.forward_step(cur, state, enc_outputs, src_mask)
            cur = logits.argmax(dim=-1)
            cur = torch.where(finished, torch.full_like(cur, PAD_ID), cur)
            outputs.append(cur)
            finished = finished | (cur == eos_id)
            if finished.all():
                break
        return torch.stack(outputs, dim=1)  # (B, L)
