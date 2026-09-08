#!/usr/bin/env python3
"""Thin OpenAI-compatible inference server for Navya checkpoints.

Loads a train.py checkpoint + the project tokenizer and serves:
  GET  /health               → model/device/params
  POST /v1/completions       → plain completion (the honest base-model API)
  POST /v1/chat/completions  → OpenAI chat shape; SFT-v2 checkpoints select the
                               embedded navya-chat-v1 token template, while
                               base/legacy checkpoints retain the Q:/A: shim.

Design: one request generates at a time (a lock — no batching; that is vLLM's
job at navya-1 scale). Full-context forward per token, no KV cache: at 57.7M
params this is ~30-60 tok/s on the A10 and ~5-15 tok/s on CPU — fine for a
research preview, deliberately not a product server.

Run:  python serve.py --ckpt ../04-training-stack/out/navya-0/ckpt_last.pt \
                      --tokenizer ../01-tokenizer/tokenizer-v0.3-64k.json \
                      --host 0.0.0.0 --port 8000
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "04-training-stack"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "07-sft"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "09-tools"))
from model import GPT, ModelConfig  # noqa: E402
import chat_format  # noqa: E402

# Tool router (three-layer architecture, layer 2): live numbers come from
# official sources, never from the model. NAVYA_TOOLS=0 disables.
tool_router = None
if os.environ.get("NAVYA_TOOLS", "1") != "0":
    try:
        import router as tool_router  # noqa: E402
    except Exception as _e:  # noqa: BLE001
        print(f"tools disabled (import failed: {_e})")

from tokenizers import Tokenizer  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "navya-0")
MAX_NEW_DEFAULT = 256
# SFT checkpoints behave best with cool sampling: they near-memorise a small
# authored corpus, and hot temperature / strong repetition penalty push
# generation OFF the memorised answer mid-sequence. Base checkpoints want the
# hotter classic settings — override via env when serving a base model.
TEMP_DEFAULT = float(os.environ.get("NAVYA_TEMP", "0.3"))
REP_PEN_DEFAULT = float(os.environ.get("NAVYA_REP_PEN", "1.1"))
GEN_LOCK = threading.Lock()

# Idle-watchdog heartbeat: the watchdog deallocates the VM when this file
# goes stale. Every served generation must touch it — for weeks nothing
# did, so the box died 15 min after every boot even under live traffic.
_HB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "last_request.txt")


def _heartbeat():
    try:
        with open(_HB_PATH, "w") as f:
            f.write(str(int(time.time())))
    except OSError:
        pass


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(ckpt_path, tokenizer_path, device):
    # Load to CPU first: a 1.31B fp32 state dict (~7.4GB) copied straight onto a
    # 12GB A10 alongside the model OOMs. Build on CPU, load the weights, then move
    # to the GPU as bf16 (halves resident size) and drop the CPU state-dict copy.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model = (model.to(device=device, dtype=torch.bfloat16)
             if device == "cuda" else model.to(device))
    ckpt.pop("model", None)
    model.eval()
    tok = Tokenizer.from_file(tokenizer_path)
    eos = tok.token_to_id("<|eos|>")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {ckpt_path} · step {ckpt.get('step')} · "
          f"{n_params/1e6:.1f}M params · {device}")
    return model, cfg, tok, eos, ckpt


@torch.no_grad()
def generate(model, cfg, tok, eos, device, prompt, max_new_tokens,
             temperature=TEMP_DEFAULT, top_k=50, top_p=0.95, stop=None,
             repetition_penalty=REP_PEN_DEFAULT, stop_ids=()):
    # prompt: text (legacy qa template) or pre-encoded ids (chat template)
    ids = list(prompt) if not isinstance(prompt, str) else tok.encode(prompt).ids
    context_budget = max(1, cfg.max_seq_len - max_new_tokens - 1)
    ids = ids[-context_budget:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -cfg.max_seq_len:])
        logits = logits[:, -1, :].float()
        if repetition_penalty > 1.0 and (out_ids or True):
            seen = torch.unique(x[0, -256:])
            sel = logits[0, seen]
            logits[0, seen] = torch.where(sel > 0, sel / repetition_penalty,
                                          sel * repetition_penalty)
        if temperature <= 0.01:
            next_id = int(logits.argmax(-1))
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                mask = sp.cumsum(-1) - sp > top_p
                sp[mask] = 0.0
                sp /= sp.sum(-1, keepdim=True)
                next_id = int(si[0, torch.multinomial(sp[0], 1)])
            else:
                next_id = int(torch.multinomial(probs[0], 1))
        if next_id == eos or next_id in stop_ids:
            break
        out_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if stop:
            text_so_far = tok.decode(out_ids)
            if any(s in text_so_far for s in stop):
                for s in stop:
                    if s in text_so_far:
                        return (text_so_far.split(s)[0], len(ids),
                                len(out_ids), time.time() - t0)
    return tok.decode(out_ids), len(ids), len(out_ids), time.time() - t0


FEW_SHOT = """The following are short, factual answers about personal finance in India.

Q: What is a SIP?
A: A SIP (Systematic Investment Plan) lets you invest a fixed amount in a mutual fund every month, which averages your purchase cost over time.

Q: What is a credit score?
A: A credit score is a number that lenders use to judge how reliably you repay debt; in India the most common one is the CIBIL score.

Q: What is an emergency fund?
A: An emergency fund is money kept safe and liquid, usually six months of expenses, so unexpected costs do not force you into debt."""


FEWSHOT_ON = os.environ.get("NAVYA_FEWSHOT", "1") == "1"
# Normally inferred from the checkpoint. The env override is retained for
# compatibility and deliberate base-model prompt experiments.
TEMPLATE_OVERRIDE = os.environ.get("NAVYA_TEMPLATE")


def render_chat(messages, tok=None, template="qa"):
    """Render messages for the model. "chat" = SFT-v2 token template (returns
    pre-encoded ids via chat_format.encode_prompt); "qa" = legacy Q:/A: text
    shim with optional few-shot priming (NAVYA_FEWSHOT=0 for SFT ckpts)."""
    if template == "chat":
        return chat_format.encode_prompt(tok, messages)
    lines = [FEW_SHOT] if FEWSHOT_ON else []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            lines.append(m["content"])
        else:
            lines.append(("Q: " if role == "user" else "A: ") + m["content"])
    lines.append("A:")
    return "\n".join(lines)


# ---- identity guard -------------------------------------------------------
# This is deliberately conservative. The previous semantic guard embedded only
# the final user turn with the model itself. On a contextual follow-up such as
# "achha, lekin phir dono alag kyun hain?" it falsely fired and replaced the
# model answer with the canned identity paragraph. A strict intent pattern is a
# better trade-off: missed paraphrases can still be answered by the model, while
# unrelated finance conversations must never be hijacked.
IDENTITY_QUERY = re.compile(
    r"(?:\b(?:who|what)\s+(?:are\s+you|is\s+your\s+name)\b|"
    r"\b(?:who\s+(?:made|built|created)\s+you|introduce\s+yourself|"
    r"tell\s+me\s+(?:about\s+yourself|your\s+name)|"
    r"what\s+model\s+are\s+you|are\s+you\s+(?:an?\s+)?(?:ai|bot|model))\b|"
    r"\b(?:tum|tu|aap)\s+(?:ho\s+)?kaun\s+(?:ho|hai|hain)\b|"
    r"\b(?:tumhara|tera|aapka)\s+naam\s+kya\s+(?:hai|hain)\b|"
    r"\b(?:tumhe|aapko)\s+kisne\s+banaya\b|"
    r"\bkhud\s+ke\s+baare\s+mein\s+batao\b|"
    r"(?:तुम|तू|आप)\s+कौन\s+(?:हो|है|हैं)|"
    r"(?:तुम्हारा|तेरा|आपका)\s+नाम\s+क्या\s+(?:है|हैं))",
    re.I,
)
IDENTITY_EN = (
    "I'm **Navya** — an India-first personal-finance AI, built from scratch by "
    "Navyam AI (Bachatt) (not fine-tuned from another model). I help with "
    "savings, SIPs & mutual funds, loans & EMIs, credit scores, insurance, and "
    "tax/GST, in English and Hinglish. I'm a research model, so I'm not a "
    "substitute for a licensed advisor — but ask me anything about money!")
IDENTITY_HI = (
    "Main **Navya** hoon — India ke liye scratch se bani ek personal-finance AI, "
    "Navyam AI (Bachatt) dwara banayi gayi (kisi aur model se fine-tune nahi ki "
    "gayi). Main savings, SIP, mutual funds, loan/EMI, credit score, insurance, "
    "aur tax/GST me madad karti hoon. Main ek research model hoon, isliye "
    "licensed advisor ka vikalp nahi — par paison se juda kuch bhi poochhiye!")
_HI_MARK = re.compile(
    r"\b(kya|kaise|kaun|kitna|hai|ho|tum|tu|aap|naam|mujhe|batao|hoon|hain|"
    r"karo|mera|meri|paisa|paise)\b|[ऀ-ॿ]", re.I)


def is_identity_query(messages):
    """Return true only for an explicit identity question in the last turn."""
    query = next((m.get("content", "") for m in reversed(messages)
                  if m.get("role") == "user"), "")
    query = " ".join(query.strip().split())
    return bool(query and len(query) <= 400 and IDENTITY_QUERY.search(query))


_THINK_BLOCK = re.compile(
    r"\A\s*<think>\s*(.*?)\s*</think>\s*(.*?)\s*\Z", re.I | re.S)


def split_reasoning_output(text):
    """Map a complete inline reasoning block to separate API fields."""
    text = text if isinstance(text, str) else ""
    match = _THINK_BLOCK.match(text)
    if not match:
        return text.strip(), ""
    return match.group(2).strip(), match.group(1).strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "navya-serve"
    ctx = {}   # filled in main()

    def _send(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            c = self.ctx
            return self._send(200, {"ok": True, "model": MODEL_ID,
                                    "device": c["device"],
                                    "step": c["step"], "params_m": c["params_m"],
                                    "chat_template": c["template"]})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "navyam"}]})
        self._send(404, {"error": "not found"})

    def _identity_answer(self, messages):
        """Return the guarded answer only for an explicit identity question."""
        if os.environ.get("NAVYA_IDENTITY_GUARD", "1") == "0":
            return None
        if not is_identity_query(messages):
            return None
        q = next(m.get("content", "") for m in reversed(messages)
                 if m.get("role") == "user")
        return IDENTITY_HI if _HI_MARK.search(q) else IDENTITY_EN

    def do_POST(self):
        path = self.path
        if path not in ("/v1/completions", "/v1/chat/completions"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            max_new = min(int(body.get("max_tokens") or MAX_NEW_DEFAULT), 512)
            temperature = float(body.get("temperature", TEMP_DEFAULT))
            top_p = float(body.get("top_p", 0.95))
            rep_pen = float(body.get("repetition_penalty", REP_PEN_DEFAULT))
            c = self.ctx
            stop = body.get("stop")
            if stop is None:
                stop = [] if c["template"] == "chat" else ["\nQ:", "\nA:"]
            if isinstance(stop, str):
                stop = [stop]
            if path == "/v1/chat/completions":
                messages = body.get("messages", [])
                ident = self._identity_answer(messages)
                if ident is not None:
                    print("identity: guarded (embedding match)")
                    return self._send(200, {
                        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
                        "object": "chat.completion",
                        "model": MODEL_ID + "+identity",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant",
                                                 "content": ident}}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0}})
                if tool_router is not None:
                    try:
                        tooled = tool_router.route(messages)
                    except Exception as te:  # noqa: BLE001
                        print(f"tool route error: {te}")
                        tooled = None
                    if tooled:
                        print("tool: answered from live data")
                        return self._send(200, {
                            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
                            "object": "chat.completion",
                            "model": MODEL_ID + "+tools",
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant",
                                                     "content": tooled}}],
                            "usage": {"prompt_tokens": 0,
                                      "completion_tokens": 0,
                                      "total_tokens": 0}})
                prompt = render_chat(messages, c["tok"], c["template"])
            else:
                prompt = str(body.get("prompt", ""))
            if isinstance(prompt, str) and not prompt.strip():
                return self._send(400, {"error": "empty prompt"})
            with GEN_LOCK:
                stop_ids = ()
                if c["template"] == "chat":
                    special = chat_format.token_ids(c["tok"])
                    stop_ids = (special["end_turn"], special["eos"])
                text, n_in, n_out, dt = generate(
                    c["model"], c["cfg"], c["tok"], c["eos"], c["device"],
                    prompt, max_new, temperature, top_k=50, top_p=top_p,
                    stop=stop, repetition_penalty=rep_pen, stop_ids=stop_ids)
            text = text.strip()
            usage = {"prompt_tokens": n_in, "completion_tokens": n_out,
                     "total_tokens": n_in + n_out}
            if path == "/v1/chat/completions":
                content, reasoning = split_reasoning_output(text)
                message = {"role": "assistant", "content": content}
                if reasoning:
                    message["reasoning_content"] = reasoning
                resp = {"id": "chatcmpl-" + uuid.uuid4().hex[:12],
                        "object": "chat.completion", "model": MODEL_ID,
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": message}],
                        "usage": usage}
            else:
                resp = {"id": "cmpl-" + uuid.uuid4().hex[:12],
                        "object": "text_completion", "model": MODEL_ID,
                        "choices": [{"index": 0, "text": text,
                                     "finish_reason": "stop"}],
                        "usage": usage}
            print(f"gen: {n_out} tok in {dt:.1f}s ({n_out/max(dt,1e-9):.0f} tok/s)")
            _heartbeat()
            self._send(200, resp)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--ckpt", default=os.path.join(
        here, "..", "04-training-stack", "out", "navya-0", "ckpt_last.pt"))
    ap.add_argument("--tokenizer", default=os.path.join(
        here, "..", "01-tokenizer", "tokenizer-v0.3-64k.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = pick_device()
    model, cfg, tok, eos, checkpoint = load(args.ckpt, args.tokenizer, device)
    declared = (checkpoint.get("chat_template") or {}).get("version")
    if declared and declared != chat_format.TEMPLATE_VERSION:
        raise ValueError(f"unsupported checkpoint chat template: {declared}")
    template = TEMPLATE_OVERRIDE or ("chat" if declared else "qa")
    if template not in ("chat", "qa"):
        raise ValueError("NAVYA_TEMPLATE must be 'chat' or 'qa'")
    Handler.ctx = {"model": model, "cfg": cfg, "tok": tok, "eos": eos,
                   "device": device, "step": checkpoint.get("step", "?"),
                   "template": template,
                   "params_m": round(sum(p.numel()
                                         for p in model.parameters()) / 1e6, 1)}
    print(f"navya inference server → http://{args.host}:{args.port}/v1")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
