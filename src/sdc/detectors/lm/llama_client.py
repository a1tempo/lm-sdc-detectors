"""Thin wrapper over llama.cpp (llama-cpp-python) for Qwen3, 4-bit GGUF.

What the LM detectors need, and how each was verified (results-log.md, 2026-11-02):
  * P(yes), P(no) for the next token after a prompt. The model answers with the tokens
    "yes" / "no" (lowercase, no leading space); " yes" / "Yes" are ~e^-10 less likely.
  * Thinking mode: generate, then read P(yes), P(no) at the first non-whitespace
    position after "</think>".
  * The final hidden state at the answer position. With embeddings on and pooling NONE,
    one decode returns logits AND this vector, and the vector is the post-final-RMSNorm
    input to the output layer.

Pinned runtime settings (they change the scores, so keep them identical for fitting,
evaluation and the Pi): flash_attn=False changes the single-pass score from 0.52 to 0.39
on the same prompt compared with flash attention on.

"Answer position" below means the position whose output distribution predicts the
answer token, i.e. the last token of the input fed to the model.
"""

import ctypes

import jinja2
import llama_cpp
import numpy as np


class QwenClient:
    def __init__(self, gguf_path: str, n_ctx: int = 2048, n_threads: int = 4,
                 flash_attn: bool = False):
        # embedding=True + pooling NONE creates a context that CAN return per-token hidden
        # states. We switch embeddings on only for calls that need them (see _decode),
        # because with embeddings on llama.cpp computes outputs for every input token
        # (~25% slower per decode on the laptop).
        # n_batch = n_ctx: _decode feeds a whole prompt in one llama_decode call, and
        # llama.cpp aborts the process if that call has more than n_batch tokens
        # (default 512; white-box reference prompts are ~510-560).
        self.llm = llama_cpp.Llama(
            model_path=gguf_path, n_ctx=n_ctx, n_batch=n_ctx, n_threads=n_threads,
            flash_attn=flash_attn, embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE, verbose=False,
        )
        self._ctx = self.llm._ctx.ctx
        self.n_ctx = n_ctx
        self.n_vocab = self.llm.n_vocab()
        self.n_embd = self.llm.n_embd()
        self._template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
            self.llm.metadata["tokenizer.chat_template"])

        # Token ids, looked up from the tokenizer and checked to be single tokens.
        self.yes_id = self._single_token("yes")
        self.no_id = self._single_token("no")
        self.think_end_id = self._single_token("</think>", special=True)

    # ---------- prompts ----------

    def chat_prompt(self, system: str, user: str, thinking: bool) -> str:
        """Render the model's own chat template. thinking=False makes the template put an
        empty <think></think> block in the prompt, so the answer is the next token."""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._template.render(messages=msgs, add_generation_prompt=True,
                                     enable_thinking=thinking)

    # ---------- detector-facing methods ----------

    def yes_no_probs(self, prompt: str) -> tuple[float, float]:
        """P(yes), P(no) for the next token after `prompt` (single-pass detector)."""
        logits, _ = self._decode(self._tokenize(prompt), start_pos=0, want_hidden=False)
        return self._yes_no(logits)

    def answer_hidden_state(self, prompt: str) -> np.ndarray:
        """Final-layer (post-RMSNorm) hidden state at the answer position, shape (n_embd,)."""
        _, h = self._decode(self._tokenize(prompt), start_pos=0, want_hidden=True)
        return h

    def yes_no_probs_and_hidden(self, prompt: str) -> tuple[float, float, np.ndarray]:
        """Both of the above from ONE decode: the shared forward pass of the single-pass
        and probe detectors."""
        logits, h = self._decode(self._tokenize(prompt), start_pos=0, want_hidden=True)
        p_yes, p_no = self._yes_no(logits)
        return p_yes, p_no, h

    def generate_then_answer(self, prompt: str, max_tokens: int = 2048
                             ) -> tuple[str, float | None, float | None]:
        """Thinking mode, greedy decoding. Generate until "</think>", skip whitespace
        tokens, then read P(yes), P(no) at the answer position.

        Returns (generated text, P(yes), P(no)), or (generated text, None, None) if the
        run is unparseable: no "</think>" within max_tokens, end of generation before
        an answer, or the most likely answer token is neither "yes" nor "no".
        Afterwards, self.last_n_generated is the number of tokens generated before the
        answer position.
        """
        self.last_n_generated = 0
        tokens = self._tokenize(prompt)
        budget = min(max_tokens, self.n_ctx - len(tokens))
        logits, _ = self._decode(tokens, start_pos=0, want_hidden=False)
        pos = len(tokens)
        generated: list[int] = []
        closed = False   # have we seen </think> yet?

        for _ in range(budget):
            nxt = int(np.argmax(logits))
            if closed and not self._is_whitespace(nxt):
                # This distribution is the answer distribution.
                text = self._detok(generated)
                if nxt not in (self.yes_id, self.no_id):
                    return text, None, None
                p_yes, p_no = self._yes_no(logits)
                return text, p_yes, p_no
            if llama_cpp.llama_vocab_is_eog(self.llm._model.vocab, nxt):
                break
            generated.append(nxt)
            self.last_n_generated = len(generated)
            if nxt == self.think_end_id:
                closed = True
            logits, _ = self._decode([nxt], start_pos=pos, want_hidden=False)
            pos += 1

        return self._detok(generated), None, None

    # ---------- internals ----------

    def _decode(self, tokens: list[int], start_pos: int, want_hidden: bool):
        """Feed `tokens` at positions start_pos.. in one batch; return (logits, hidden)
        at the last token. start_pos=0 clears the KV cache first (new prompt)."""
        if start_pos == 0:
            llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(self._ctx), True)
        llama_cpp.llama_set_embeddings(self._ctx, want_hidden)
        n = len(tokens)
        batch = llama_cpp.llama_batch_init(n, 0, 1)
        try:
            for i, t in enumerate(tokens):
                batch.token[i] = t
                batch.pos[i] = start_pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = (i == n - 1)   # request output only at the last token
            batch.n_tokens = n
            rc = llama_cpp.llama_decode(self._ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode returned {rc}")
        finally:
            llama_cpp.llama_batch_free(batch)

        logits = np.ctypeslib.as_array(
            llama_cpp.llama_get_logits_ith(self._ctx, -1), shape=(self.n_vocab,)).copy()
        hidden = None
        if want_hidden:
            hidden = np.ctypeslib.as_array(
                llama_cpp.llama_get_embeddings_ith(self._ctx, -1), shape=(self.n_embd,)).copy()
        return logits, hidden

    def _yes_no(self, logits: np.ndarray) -> tuple[float, float]:
        """Full-vocabulary softmax, then the two answer probabilities (float64)."""
        z = logits.astype(np.float64)
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        return float(p[self.yes_id]), float(p[self.no_id])

    def _tokenize(self, text: str) -> list[int]:
        # special=True so "<|im_start|>", "<think>" etc. in the rendered template become
        # their control tokens. Qwen3 adds no BOS token.
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def _single_token(self, text: str, special: bool = False) -> int:
        ids = self.llm.tokenize(text.encode(), add_bos=False, special=special)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token: {ids}")
        return ids[0]

    def _detok(self, ids: list[int]) -> str:
        return self.llm.detokenize(ids, special=True).decode("utf-8", errors="replace")

    def _is_whitespace(self, tok: int) -> bool:
        return self._detok([tok]).strip() == ""
