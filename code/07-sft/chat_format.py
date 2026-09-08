#!/usr/bin/env python3
"""Versioned Navya chat encoding shared by data building and serving.

Tokenizer v0.3 reserved 64 stable special-token slots precisely for additions
like chat roles. SFT v2 aliases the first five slots without changing the
tokenizer vocabulary or any existing checkpoint:

    reserved_0 = system        reserved_3 = tool
    reserved_1 = user          reserved_4 = end_turn
    reserved_2 = assistant

Only assistant content, its end-turn marker, and the final EOS are supervised.
System/user/tool text and all role markers are context-only. Encoding content
separately from structural tokens keeps response-mask boundaries exact.
"""

TEMPLATE_VERSION = "navya-chat-v1"

ROLE_TOKENS = {
    "system": "<|reserved_0|>",
    "user": "<|reserved_1|>",
    "assistant": "<|reserved_2|>",
    "tool": "<|reserved_3|>",
}
END_TURN_TOKEN = "<|reserved_4|>"
EXPECTED_TOKEN_IDS = {
    "system": 3,
    "user": 4,
    "assistant": 5,
    "tool": 6,
    "end_turn": 7,
    "bos": 0,
    "eos": 1,
    "pad": 2,
}
STRUCTURAL_TOKENS = tuple(ROLE_TOKENS.values()) + (END_TURN_TOKEN,
                                                   "<|bos|>",
                                                   "<|eos|>",
                                                   "<|pad|>")


def _single_token_id(tok, text):
    token_id = tok.token_to_id(text)
    if token_id is None or tok.encode(text).ids != [token_id]:
        raise ValueError(f"chat structural marker is not one token: {text}")
    return token_id


def token_ids(tok):
    """Resolve and validate the stable tokenizer-v0.3 chat aliases."""
    resolved = {role: _single_token_id(tok, marker)
                for role, marker in ROLE_TOKENS.items()}
    resolved.update({
        "end_turn": _single_token_id(tok, END_TURN_TOKEN),
        "bos": _single_token_id(tok, "<|bos|>"),
        "eos": _single_token_id(tok, "<|eos|>"),
        "pad": _single_token_id(tok, "<|pad|>"),
    })
    if resolved != EXPECTED_TOKEN_IDS:
        raise ValueError(f"unexpected chat token ids: {resolved}")
    return resolved


def template_metadata(tok):
    ids = token_ids(tok)
    return {
        "version": TEMPLATE_VERSION,
        "role_tokens": {
            role: {"token": marker, "id": ids[role]}
            for role, marker in ROLE_TOKENS.items()
        },
        "end_turn": {"token": END_TURN_TOKEN, "id": ids["end_turn"]},
        "bos_id": ids["bos"],
        "eos_id": ids["eos"],
        "pad_id": ids["pad"],
        "loss": "assistant_content_plus_end_turn_and_final_eos",
    }


def _normalise_messages(messages, system=None):
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    out = []
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("system must be a non-empty string")
        out.append({"role": "system", "content": system.strip()})
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_TOKENS:
            raise ValueError(f"unsupported chat role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{role} message content must be a non-empty string")
        out.append({"role": role, "content": content.strip()})
    return out


def validate_messages(messages, system=None, require_final_assistant=True):
    """Validate role ordering and return a normalised copy."""
    out = _normalise_messages(messages, system)
    if not out:
        raise ValueError("conversation is empty")
    system_positions = [i for i, m in enumerate(out) if m["role"] == "system"]
    if len(system_positions) > 1 or (system_positions and system_positions[0] != 0):
        raise ValueError("system role may appear once, at the start")
    non_system = out[1:] if system_positions else out
    if not non_system or non_system[0]["role"] != "user":
        raise ValueError("conversation must begin with a user message")
    previous = None
    for message in non_system:
        role = message["role"]
        if role == "user" and previous not in (None, "assistant", "tool"):
            raise ValueError("user must follow assistant/tool or start the chat")
        if role == "assistant" and previous not in ("user", "tool"):
            raise ValueError("assistant must follow user or tool")
        if role == "tool" and previous != "assistant":
            raise ValueError("tool result must follow an assistant message")
        previous = role
    if require_final_assistant and previous != "assistant":
        raise ValueError("training conversation must end with assistant")
    return out


def _content_ids(tok, content):
    # Prevent untrusted/user-authored text from injecting a structural marker.
    for marker in STRUCTURAL_TOKENS:
        content = content.replace(marker, marker.replace("<|", "< |", 1))
    return tok.encode(content).ids


def encode_conversation(tok, messages, system=None):
    """Return ``(token_ids, loss_mask)`` for one complete conversation."""
    messages = validate_messages(messages, system, require_final_assistant=True)
    special = token_ids(tok)
    ids = [special["bos"]]
    loss_mask = [0]
    for message in messages:
        role = message["role"]
        ids.append(special[role])
        loss_mask.append(0)
        content = _content_ids(tok, message["content"])
        trained = role == "assistant"
        ids.extend(content)
        loss_mask.extend([int(trained)] * len(content))
        ids.append(special["end_turn"])
        loss_mask.append(int(trained))
    ids.append(special["eos"])
    loss_mask.append(1)  # final message is guaranteed to be assistant
    return ids, loss_mask


def encode_prompt(tok, messages, system=None):
    """Encode chat history plus the assistant role that starts generation."""
    messages = validate_messages(messages, system, require_final_assistant=False)
    if messages[-1]["role"] not in ("user", "tool"):
        raise ValueError("generation prompt must end with user or tool")
    special = token_ids(tok)
    ids = [special["bos"]]
    for message in messages:
        ids.append(special[message["role"]])
        ids.extend(_content_ids(tok, message["content"]))
        ids.append(special["end_turn"])
    ids.append(special["assistant"])
    return ids
