import argparse
from pathlib import Path
import tiktoken
import torch
import torch.nn as nn

# 학습 때 사용한 모델 상수 (o200k_base 사용 시 vocab_size는 tokenizer.n_vocab으로 자동 동기화)
MODEL_CONFIG = {
    "vocab_size": 200019,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어떨어져야 합니다."
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def load_model_from_checkpoint(ckpt_path: Path, cfg: dict, device: torch.device) -> GPTModel:
    model = GPTModel(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        extra_info = {
            k: v
            for k, v in checkpoint.items()
            if k not in ("model_state_dict", "optimizer_state_dict")
        }
        if extra_info:
            print(f"[체크포인트 메타정보] {extra_info}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_text(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    device: torch.device,
    context_length: int,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    eot_id = tokenizer.eot_token
    input_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    if not input_ids:
        raise ValueError("프롬프트가 비어 있거나 토큰화 결과가 없습니다.")

    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(
                    logits < top_values[:, [-1]],
                    torch.tensor(float("-inf"), device=device, dtype=logits.dtype),
                    logits,
                )
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eot_id:
            break

        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx.squeeze(0).tolist())


def format_prompt(raw_prompt: str) -> str:
    cleaned = raw_prompt.replace("\\n", "\n").strip()
    if cleaned.startswith("질문:"):
        if "\n답변:" not in cleaned:
            return f"{cleaned}\n답변:"
        return cleaned
    return f"질문: {cleaned}\n답변:"


def main():
    parser = argparse.ArgumentParser(description="o200k_base Q&A 체크포인트 추론")
    parser.add_argument("--ckpt", type=str, default="model_qa_epoch_1.pt", help="불러올 .pt 체크포인트 경로")
    parser.add_argument("--encoding", type=str, default="o200k_base", help="tiktoken 인코딩 이름")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.8, help="샘플링 온도 (0이면 탐욕적 생성)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K 샘플링 값")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path.resolve()}")

    tokenizer = tiktoken.get_encoding(args.encoding)
    MODEL_CONFIG["vocab_size"] = tokenizer.n_vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"실행 장치: {device}")
    print(f"체크포인트 로드 중: {ckpt_path}")

    model = load_model_from_checkpoint(ckpt_path, MODEL_CONFIG, device)
    print("\n모델 로드 완료! 질문을 입력하세요 (종료: 'quit' 또는 빈 줄 Enter)")

    test_count = 0
    while True:
        try:
            raw_prompt = input("\n질문 입력 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not raw_prompt or raw_prompt.lower() in {"quit", "exit"}:
            print("종료합니다.")
            break

        prompt = format_prompt(raw_prompt)
        test_count += 1

        print("\n" + "=" * 60)
        print(f"[테스트 {test_count}]")
        print(f"전달된 프롬프트:\n{prompt}")
        print("-" * 60)

        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            context_length=MODEL_CONFIG["context_length"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        generated_only = output[len(prompt):] if output.startswith(prompt) else output
        print(f"모델 답변:\n{generated_only.strip()}")
        print("=" * 60)


if __name__ == "__main__":
    main()