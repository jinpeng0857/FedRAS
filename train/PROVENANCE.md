# Federated supplement provenance

The FedDR supplement roster is implemented as a thin adapter around the
existing FEDEATA data split, model, checkpoint, and logging interfaces:

- FedAvg: standard sample-count-weighted aggregation;
- FedProx: existing client proximal objective;
- SCAFFOLD: Option-II server/client control variates;
- FedDyn: dynamic local regularization and server history update;
- FedAdam: adaptive server moments over aggregated client deltas.

These algorithms only produce federated initialization checkpoints. They do
not replace the primary FedDR TTA adapter, which remains
`FedACG + DLRS + FSLR` in `main_test.py`.

