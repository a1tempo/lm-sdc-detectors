from sdc.detectors.classical.range_check import RangeCheck


class QuantileMonitor(RangeCheck):
    """Like RangeCheck, but the bounds for layer l are the 0.1% and 99.9% quantiles
    of its per-run min and max across clean fitting runs, which makes them less
    sensitive to a single extreme clean run. Needs no fields beyond LayerStats.

    Listed in the planning notes but not in the proposal.
    TODO: confirm this definition with the team and add it to the paper.
    """
    name = "quantile"
    low_key, high_key = "quantile_low", "quantile_high"
