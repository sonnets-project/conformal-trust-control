# Third-party dependencies

The empirical episode protocol follows
[Empirical Deep Hedging](https://github.com/oskarimikkila/Empirical-Deep-Hedging)
by Oskari Mikkila and Juho Kanniainen. The original experiments used an adapted
version of that environment. This repository implements the contract-path
selection and BS benchmark needed by the residual-policy experiments in
`ctc_runtime.py`; it does not distribute the upstream `include/` modules.

NumPy, pandas, SciPy, Matplotlib, PyTorch, QuantLib, scikit-learn, joblib,
PyArrow, yfinance and Jupyter are installed separately under their own licenses.
The training requirements retain the dependencies of the original environment,
including libraries used by its simulation and model-loading utilities.
The [MIT license](LICENSE) applies to this project's original code.
