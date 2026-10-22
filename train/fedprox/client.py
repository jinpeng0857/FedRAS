"""FedProx client: local SGD with a proximal global-model penalty."""

from train.fedacg.client import FedACGClient


class FedProxClient(FedACGClient):
    """Use the shared local engine with the canonical FedProx objective."""

    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        self.ce_weight = 1.0
        self.prox_weight = float(getattr(configs, "fedprox_mu", 0.01))
        if self.prox_weight <= 0:
            raise ValueError("--fedprox_mu must be positive for FedProx")
