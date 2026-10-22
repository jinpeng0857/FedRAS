"""FedAvg client implemented as standard local cross-entropy SGD."""

from train.fedacg.client import FedACGClient


class FedAvgClient(FedACGClient):
    """Reuse the shared local-SGD engine while disabling FedACG-only proximal loss."""

    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        self.ce_weight = 1.0
        self.prox_weight = 0.0
