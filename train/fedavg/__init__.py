"""FedAvg federated-training implementation."""

from train.fedavg.client import FedAvgClient
from train.fedavg.server import FedAvgServer

__all__ = ["FedAvgClient", "FedAvgServer"]
