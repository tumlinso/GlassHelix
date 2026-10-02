# Native observation consumer

`NativeProductReadout` lowers identified linear contributions `k*h` to installed
Cellerator product2 packets `k*h*1`. A Torch `index_add` sums contributions into
explicit observation slots. Repeated source/observation indices retain separate
contribution identities and coefficients. CPU FP32 forward, input/coefficient
VJP and the full readout JVP are qualified. Torch owns optimization; checkpoint
restore requires the same declared identities and topology.

`examples.py` uses the accepted MatrixPatchField for a short trajectory, then
trains an actual native readout. It also applies a supplied coordinate
refactoring and adds anonymous ResidualActor features with zero initial native
outgoing coefficients. Those coefficients have live initial gradients; hidden
feature gradients become live after the first update. Existing field and
refactoring calculations remain Torch. This is equivalent selected readout
lowering, with synthetic observations and biology `not_run`. There is no
performance claim, automatic identity discovery, or Torch CUDA qualification.

Run the small actual consumer gate from the repository root:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/integration/check.py
```

The gate checks the existing receipt/install manifest, exact Cellerator commit,
installed package bytes and native library hash before importing the installed
adapter from `/tmp/gh-moon-celleratorch`. It never changes ML2 site-packages.
Eight check groups cover independent forward/VJP/JVP references, finite
differences, native coefficient learning, Adam checkpoint continuation,
identity/topology rejection, trajectory gradients and refactoring/regrowth.
A normal run leaves tracked evidence unchanged and checks recorded source
hashes. `--record` explicitly records a fresh run in
`evidence/consumer-checks.json`; use it after an intentional source change.
