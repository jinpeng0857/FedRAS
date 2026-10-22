"""FedProx federated-training implementation."""

from train.fedprox.client import FedProxClient
from train.fedprox.server import FedProxServer

__all__ = ["FedProxClient", "FedProxServer"]
