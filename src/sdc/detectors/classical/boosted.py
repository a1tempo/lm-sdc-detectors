from sklearn.ensemble import HistGradientBoostingClassifier

from sdc.detectors.classical.learned import LearnedDetector


class BoostedDetector(LearnedDetector):
    """Gradient-boosted decision trees on the raw observation features (non-linear baseline).

    Same features and training data as LearnedDetector; only the classifier changes. Trees
    can flag both ends of a feature (e.g. an SDC that is over-confident and one that is
    unsure), which a linear model on (top1_prob, entropy) cannot. It answers whether the
    LM probe's advantage in black-box comes from the LM or just from being non-linear.
    Defaults of sklearn's HistGradientBoostingClassifier, with a fixed seed.
    """
    name = "boosted"

    def __init__(self, access_level: str = "whitebox"):
        super().__init__(access_level)
        self.model = HistGradientBoostingClassifier(random_state=0)
