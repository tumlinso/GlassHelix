# Plain one-cell Torch references

`model.py` implements the agreed `MatrixPatchField` and `SylvesterFlow` exports. These independent ordinary prediction families retain the immutable seed algebra. Patches are internal state layout; latent coordinates carry no identified biological meaning.

`MatrixPatchField(patches, width=16)` accepts exactly `(patches,width,width)` and returns that shape. Trainable left/right maps and softplus decay remain connected to autograd. `SylvesterFlow(width=16)` accepts exactly `(width,width)` with finite scalar `dt` and evaluates `exp(dt L) X exp(dt R)`. This is the flow formula for fixed autonomous `dX/dt = L X + X R`; numerical matrix exponentials introduce floating point error. It does not solve nonlinear fields or changing generators. Each call recomputes exponentials, so a parameter update cannot silently reuse a stale detached cache.

Inputs must match parameter dtype/device; float32 and float64 are supported, including strided inputs. Initial modules use Torch's default floating dtype/device and can be converted with `.double()` or `.to(...)`. A tensor `dt` must be scalar and match input dtype/device; its gradient is preserved. CPU float32/float64 are qualified here. Other devices remain unqualified.

Run the recorded checks from the checkout root:

```sh
CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/plain/validate.py
```

The seven checks compare seed algebra against per-patch contractions, finite-difference all input and parameter derivatives, finite-difference flow time derivatives, verify fixed-generator composition and the instantaneous generator, exercise shape/dtype rejection, and fit both families to small synthetic teacher outputs. `validation.json` records observed losses, source hashes and runtime. Parameter estimates need not be unique; convergence concerns predictions on the supplied synthetic inputs.

This delivers reference models and engineering checks. Native adapters, biological evidence, CUDA qualification and throughput measurements remain subsequent owner tasks. Existing scientific candidates and ML2 obligations retain their own evidence and workflow.
