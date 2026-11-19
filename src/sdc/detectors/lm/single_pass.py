from sdc.detectors.lm.common import LMDetector


class SinglePassDetector(LMDetector):
    """Non-thinking mode, one forward pass: Eq. 2 with r_i omitted.

    The chat template puts an empty <think></think> block in the prompt, so the answer
    is the very next token. The score always exists; `yes_no_mass` = P(yes) + P(no)
    shows how much of the next-token probability the two answers hold.

    No trained decision head, so the score need not be calibrated for detection.
    """
    name = "lm_single_pass"
    thinking = False

    def _query(self, prompt: str) -> dict:
        p_yes, p_no = self.client.yes_no_probs(prompt)
        return {"p_yes": p_yes, "p_no": p_no, "parsed": True, "yes_no_mass": p_yes + p_no}
