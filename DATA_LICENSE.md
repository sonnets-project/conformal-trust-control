# Data availability

The OptionMetrics IvyDB data were accessed through Imperial College London
under an institutional subscription. The agreement does not permit
redistribution. Raw quotes, the cleaned panel and contract-level records
are not included in this repository.

`results/reference/` contains aggregate statistics reported in the paper.
The cleaning code and input specifications are provided in
[data/README.md](data/README.md).

`pretrained/` contains 120 residual-policy checkpoints: network weights and
scalar training settings. It contains no quote tables or episode records.
Evaluation creates those records locally from the reader's own data.

The [MIT license](LICENSE) covers the code and supplied model weights. OptionMetrics,
index-price and Treasury inputs remain subject to their providers' terms.
