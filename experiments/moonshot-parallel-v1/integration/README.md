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

Run the installed consumer gate from a Python environment containing Cellerator
and its optional Torch integration:

```sh
python -m pip install "/home/tumlinson/Cellerator[torch]"
python -B experiments/moonshot-parallel-v1/integration/check.py --record
```

The gate discovers the installed `cellerator` distribution through Python
metadata, checks its packaged native files, and imports product2 through
`cellerator.torch`. It records fresh consumer evidence separately at
`evidence/consumer-checks-cellerator.json`, leaving the older component-era
receipt untouched. Eight check groups cover independent forward/VJP/JVP
references, finite differences, native coefficient learning, Adam checkpoint
continuation, identity/topology rejection, trajectory gradients and
refactoring/regrowth. A normal run requires matching current evidence and checks
that the GH sources and installed package files have not changed.
