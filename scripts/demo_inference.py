"""Quick batch demo of the trained debate model — useful for PPT screenshots.

Usage:
    PYTHONPATH= CUDA_VISIBLE_DEVICES=3 uv run python scripts/demo_inference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
import torch

from src.data import EOS_ID, PAD_ID, SOS_ID
from src.generate import beam_search, detok
from src.model.lstm_seq2seq import Seq2Seq

CKPT = Path("checkpoints/xlmr-1777348083/best.pt")
SP = Path("data/processed/spm.model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLES = [
    # (topic / motion, input_context PRO argument)
    ("안락사 허용",
     "고통스러운 삶을 강제로 이어가게 하는 것은 비인도적입니다."),
    ("재택근무는 직장 내 생산성 향상에 기여한다.",
     "재택근무는 직원에게 자율성을 주어 업무 몰입도를 높이고 출퇴근 스트레스를 줄여준다."),
    ("자전거 전용도로 확충은 도시 교통 문제 해결에 가장 효과적인 방안이다.",
     "자전거 전용도로를 늘리면 친환경 이동수단 이용이 증가해 차량 통행량과 교통 혼잡이 줄어든다."),
    ("Legalization of Euthanasia",
     "It is inhumane to force someone to continue a life full of suffering."),
    ("We should ban single-use plastics.",
     "Single-use plastics are a major source of ocean pollution and harm wildlife."),
    ("정기적인 건강검진은 모든 성인에게 의무화되어야 한다.",
     "조기에 질병을 발견하면 의료비를 절감하고 국민 건강 수준을 향상시킬 수 있다."),
]


def main() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(SP))

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    train_args = ckpt["args"]

    model = Seq2Seq(
        vocab_size=ckpt["vocab_size"],
        embed_dim=train_args.get("embed_dim", 256),
        hidden_dim=train_args.get("hidden_dim", 512),
        enc_layers=train_args.get("enc_layers", 2),
        dec_layers=train_args.get("dec_layers", 1),
        dropout=0.0,
        encoder_type="xlmr",
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    print("=" * 78)
    print(f"  Debate Battle LSTM Seq2Seq — checkpoint: {CKPT}")
    print(f"  Encoder: frozen XLM-RoBERTa-base | Decoder: LSTM + attention")
    print(f"  Test BLEU: {json.loads(Path(CKPT.parent / 'test_metrics.json').read_text())['bleu']:.3f}")
    print("=" * 78)

    for i, (topic, input_ctx) in enumerate(EXAMPLES, 1):
        encoder_input = f"{topic} <SEP> {input_ctx}"
        enc = tok(encoder_input, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = beam_search(
                model,
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                beam_size=4,
                max_len=64,
                length_penalty=1.0,
            )
        output = detok(sp, out_ids)
        print(f"\n[{i}] TOPIC : {topic}")
        print(f"    PRO   : {input_ctx}")
        print(f"    CON ↓ : {output}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
