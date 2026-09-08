# SFT trainer v2

`chat_format.py` is the versioned template source of truth. It maps tokenizer
reserved IDs 3–7 to system, user, assistant, tool, and end-turn semantics.
`build_chat_data.py` creates packed token/mask/example arrays plus a provenance
manifest. `sft_train.py` applies assistant-only loss, conversation-isolated
attention, per-conversation RoPE reset, deterministic row coverage, and exact
resume. `serve.py` reads the template metadata embedded in the checkpoint.

The current 295-conversation corpus is for correctness and style experiments.
The production config deliberately gates training at 10,000 reviewed training
conversations; it must not be weakened merely to produce a checkpoint.

Run the regression suite from the repository root:

```bash
.venv/bin/python -m unittest discover -s code/07-sft -p 'test_*.py' -v
```
