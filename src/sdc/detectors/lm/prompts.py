"""Prompt text for the LM detectors.

Two context conditions (proposal §6.2):
  zero_shot: serialized observation only
  reference: observation + per-layer fault-free ranges from the fitting set
Thinking vs. non-thinking is not decided here: QwenClient.chat_prompt renders the
model's chat template with enable_thinking on or off.
"""

TASK = (
    "You monitor a ResNet-18 image classifier running on an Earth-observation satellite. "
    "A hardware fault, a single flipped bit in a weight or an activation, may have hit "
    "this inference. Your job is to decide whether it silently changed the predicted class."
)

DESCRIBE_BLACKBOX = (
    "You see the classifier's output: the top-1 softmax probability and the entropy of "
    "the output distribution in nats."
)

DESCRIBE_WHITEBOX = (
    "You see activation statistics at six points in the network (stem; the four residual "
    "stages layer1 to layer4; fc, the output logits): min, max, mean, standard deviation, "
    "and the count of non-finite values. Then the classifier's output: the top-1 softmax "
    "probability and the entropy of the output distribution in nats."
)

QUESTION = "Did a fault change this run's predicted class? Answer with exactly one word: yes or no."


def build_messages(observation_text: str, access_level: str,
                   reference_text: str | None = None) -> tuple[str, str]:
    """Return (system, user) messages. Pass reference_text for the "reference" condition."""
    describe = {"blackbox": DESCRIBE_BLACKBOX, "whitebox": DESCRIBE_WHITEBOX}[access_level]
    system = f"{TASK} {describe}"
    parts = []
    if reference_text is not None:
        parts.append(f"Normal ranges, from fault-free runs:\n{reference_text}")
    parts.append(f"This run:\n{observation_text}")
    parts.append(QUESTION)
    return system, "\n\n".join(parts)
