"""FedProx server: standard sample-weighted averaging of proximal local SGD."""

import copy

from train.fedavg.server import FedAvgServer


class FedProxServer(FedAvgServer):
    """FedProx changes the client objective, not the FedAvg aggregation rule."""

    def _build_checkpoint(self, round_num, final=False):
        del final
        model_state = copy.deepcopy(self.backbone.state_dict())
        return {
            "round": round_num,
            "epoch": round_num,
            "method": "fedprox",
            "model_state_dict": copy.deepcopy(model_state),
            "adaptation_anchor": model_state,
            "anchor_type": "fedprox_global_model",
            "best_clean_acc": self.best_clean_acc,
            "best_clean_round": self.best_clean_round,
            "config_info": {
                "method": "fedprox",
                "fedprox_mu": getattr(self.configs, "fedprox_mu", None),
                "num_class": getattr(self.configs, "num_class", None),
                "backbone": getattr(self.configs, "backbone", None),
                "dataset": getattr(self.configs, "dataset", None),
                "alpha": getattr(self.configs, "alpha", None),
            },
        }
