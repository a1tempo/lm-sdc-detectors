from sdc.detectors.lm.common import LMDetector


class GenerativeDetector(LMDetector):
    """Thinking mode: generate reasoning r_i, then
    s_i = P(yes|o_i,r_i) / (P(yes|o_i,r_i) + P(no|o_i,r_i))   (proposal Eq. 2).

    Decoding is greedy, so a run's reasoning is reproducible. Runs with no parseable
    answer (no </think> within the token budget, generation ended, or the most likely
    answer token is neither "yes" nor "no") get s_i = 0.5 and parsed=False, so the
    no-answer rate can be reported.
    """
    name = "lm_generative"
    thinking = True

    def __init__(self, client, condition: str = "zero_shot", access_level: str = "whitebox",
                 max_tokens: int = 2048):
        super().__init__(client, condition, access_level)
        self.max_tokens = max_tokens   # the context size n_ctx also caps generation

    def _query(self, prompt: str) -> dict:
        reasoning, p_yes, p_no = self.client.generate_then_answer(prompt, self.max_tokens)
        return {"p_yes": p_yes, "p_no": p_no, "parsed": p_yes is not None,
                "n_gen_tokens": self.client.last_n_generated, "reasoning": reasoning}
