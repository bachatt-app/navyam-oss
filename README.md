# Bachatt — Indian Financial Foundation Model

**North star:** build a genuinely new AI model, in India, for India — trained from random
initialization, not fine-tuned on top of someone else's weights.

This repository holds the **open-source code** for that effort — the tokenizer, data
pipeline, training stack, evaluation harness, and inference server used to build the Navya
family of India-first financial language models.

> **Models & weights** are released separately on
> [Hugging Face](https://huggingface.co/navyam-ai) and
> [Ollama](https://ollama.com/navyam-ai) (`ollama run navyam-ai/navya-1`). Training data and
> internal planning documents are not part of this repository.

---

## 1. Objective

Train the core language model **from random initialization**, while being pragmatic about
using existing open models as:

- teachers (distillation)
- evaluators (LLM-as-judge)
- synthetic-data generators
- research baselines

**Modality scope (v1): text only.** Voice and vision come later — they are explicitly out of
scope until the text model is strong.

**Capability priority:**

1. **Finance** — savings, investment, stocks, cards, loans, insurance, tax (the superpower)
2. **Coding** — the next natural extension, but in pretraining from day one
3. **General assistance** — ChatGPT-class helpfulness as table stakes

The goal is not "an AI that knows India well." It is:

> **world-class intelligence + unusually deep Indian knowledge.**

---

## 2. Model progression

Rather than immediately attempting one enormous 400B model, develop a generational ladder.
The architecture and training stack are designed from day one so the **same infrastructure
scales upward** — no rewrite between generations.

| Generation | Model                       | Purpose                                              |
| ---------- | --------------------------- | ---------------------------------------------------- |
| Research   | 300M–1B                     | Validate tokenizer, data pipeline, architecture       |
| Alpha      | 7–8B                        | First genuinely useful India/finance model            |
| Beta       | 30–40B                      | Strong finance + coding + reasoning                   |
| Flagship   | 70–100B dense / MoE equiv.  | Serious general assistant                             |
| Later      | 200B+ MoE                   | Only after proving scaling and data quality           |

**The first model targeted to be commercially impressive is ~30B** — not 1B, and not 400B.
Everything below 7B is research infrastructure, not product.

---

## 3. Pretraining corpus — India-first, not India-only

Approximate starting mix (these proportions change through experimentation; they are not
fixed forever):

| Share   | Component                     |
| ------- | ----------------------------- |
| 45–55%  | High-quality global English   |
| 15–20%  | Indian English                |
| 10–15%  | Indian languages              |
| 10–15%  | Code                          |
| 5–10%   | Finance / economics / law     |
| 2–5%    | Mathematics + reasoning       |

**Why so much global English?** Because an AI that knows India extremely well but is
intellectually weaker than general-purpose models is not worth building. Global English is
what buys raw reasoning capability; the India-specific slices are what make it uniquely ours.

---

## 4. Tokenizer

Build our own tokenizer. Minimum target languages:

English · Hindi · Hinglish · Bengali · Marathi · Telugu · Tamil · Gujarati ·
Kannada · Malayalam · Punjabi · Odia · Assamese

Urdu and additional languages come later.

### Beyond public corpora

AI4Bharat provides useful corpora, parallel data, language-processing tools and benchmarks
that can form part of the research foundation — Samanantar alone carries tens of millions of
English–Indic parallel sentence pairs.

But the real investment goes into what public datasets **do not** capture — natural
code-mixed Indian financial speech:

- "Salary kab credit hoga?"
- "Mera CIBIL 680 hai, home loan milega?"
- "₹25k SIP karoon ya loan prepay?"
- "HDFC Regalia vs SBI Cashback kaunsa better?"
- "Mujhe 80C mein kya karna chahiye?"
- "NIFTY gir raha hai, portfolio sell karoon kya?"

The model must understand code-mixing **natively**, not by mentally translating through
English. This matters enormously once voice is added.

---

## 5. Finance as the superpower — three layers

Do not try to put all financial knowledge into the weights. Three distinct layers:

### Layer 1 — Foundational financial understanding (in the weights)

Train on historical material, subject to licensing, copyright and permitted use:

- **Regulators:** RBI, SEBI, IRDAI, PFRDA publications
- **Law:** Companies Act, Income Tax material, GST material
- **Markets:** NSE/BSE material, annual reports, prospectuses, earnings transcripts,
  shareholder letters
- **Products:** mutual-fund documents, insurance policy documents, loan terms,
  credit-card terms, banking documentation
- **Macro & scholarship:** economic surveys, Union Budgets, RBI monetary-policy documents,
  research papers, financial textbooks, Indian financial journalism

The model should understand — deeply, not superficially — CAGR vs XIRR, TER, duration, yield
curve, NPAs, CRR, SLR, repo rates, LTV, EMI amortization, credit scores, ULIPs, term
insurance, health insurance, taxation, ESOPs, capital gains, dividends, corporate actions,
derivatives, mutual funds, ETFs, REITs, InvITs, EPF, PPF, NPS.

### Layer 2 — Live financial intelligence (never in the weights)

Model weights must **never** be expected to know today's stock price, NAV, interest rates,
credit-card offers, insurance premium, FD rates, tax rules, RBI circulars or SEBI rules.
These arrive through tools / APIs / RAG.

Target architecture:

```
                YOUR FOUNDATION MODEL
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Reasoning       Search         Coding
          │              │              │
          ↓              ↓              ↓
      Finance tools  Knowledge DB    Sandbox
          │
   ┌──────┼───────────┬──────────┐
   ↓      ↓           ↓          ↓
 Markets Banking  Insurance     Tax
```

This is both safer and more capable than trying to make a frozen LLM contain every answer.

### Layer 3 — Indian financial reasoning (the differentiator)

Build hundreds of thousands to millions of scenarios, e.g.:

> A 28-year-old earns ₹1.4 lakh/month, has ₹8 lakh cash, ₹12 lakh in equity MF, an ₹18 lakh
> education loan at 9.2%, wants a home in 4 years, and supports parents.
> What should they consider?

Train the model to reason across cash-flow, emergency funds, insurance, tax, debt, risk,
goals, asset allocation, opportunity cost, scenario analysis and uncertainty — **rather than
emitting "Buy XYZ fund."**

That distinction matters for both quality and regulation.

### Compliance is an engineering discipline, not a launch checklist

Financial-product deployment requires serious compliance engineering. RBI's Digital Lending
Directions carry requirements on customer protection, disclosures, and data
collection/use/sharing/storage; securities advice sits within SEBI's framework.

**Hire finance and regulatory experts alongside ML researchers from the beginning — not
after launch.**

---

## 6. Coding in pretraining from day one

Coding is not a later bolt-on. It goes into v1 pretraining at roughly **10–15%**, because
code teaches the model:

- structured reasoning
- syntax
- long-range dependencies
- tool construction
- debugging
- algorithmic thinking

Finance + coding together is where this becomes genuinely powerful. Given "Analyze my equity
portfolio," the model should autonomously generate Python:

```python
returns = prices.pct_change()
beta = covariance(stock, index) / variance(index)
portfolio_volatility = ...
max_drawdown = ...
sector_exposure = ...
```

execute it in a sandbox, and explain the result in Hindi.

That is a far more interesting target than another chatbot.

---

## 7. Standing constraints

These hold unless deliberately revised:

1. **Train from scratch.** Open models are teachers, judges, data generators and baselines —
   not the base weights.
2. **Text only for now.** Voice and vision are deferred, but architecture choices should not
   preclude them.
3. **~30B is the first commercial target.** Smaller runs exist to de-risk it.
4. **India-first, not India-only.** Never trade away general intelligence for local flavour.
5. **Live data never lives in the weights.** Tools, APIs and RAG own the present tense.
6. **Reason, don't recommend.** Financial output is structured reasoning under uncertainty,
   not product picks.
7. **Same stack scales up.** Every infrastructure decision is judged at flagship scale.

---

## Repository layout

| Path                        | Description                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `code/01-tokenizer/`        | Custom 64k/96k tokenizers — training, normalization, fertility evaluation.          |
| `code/02-data-pipeline/`    | Corpus ingestion, cleaning, deduplication, snapshots.                               |
| `code/03-evals/`            | BachattBench harness and grading.                                                   |
| `code/04-training-stack/`   | The from-scratch model (`model.py`) and training stack; run configs.                |
| `code/06-inference/`        | Inference server (`serve.py`) and checkpoint→HF/GGUF export.                         |
| `code/07-sft/`              | Supervised fine-tuning, chat template, calculators.                                  |
| `code/08-trl/`, `13-reasoning/` | DPO/GRPO and reasoning-distillation trainers.                                   |
| `infra.env.example`         | Template for cloud identifiers; copy to a local (gitignored) `infra.env`.            |
| `LICENSE`                   | Apache-2.0.                                                                          |

Start at `code/README.md`.

**Not included:** training/evaluation datasets, model weights (released on
[Hugging Face](https://huggingface.co/navyam-ai) / [Ollama](https://ollama.com/navyam-ai)),
and internal planning documents.

*Objective recorded 14 August 2026.*
