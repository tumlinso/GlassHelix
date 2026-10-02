# Installed fitted native bridge

`tests/learning/native_bridge` is a separate C++ executable linked through the
installed `GlassHelix::core` package and published Cellerator SDK targets. It
does not link Torch. Cellerator host local arithmetic computes the supplied
scalar law `r = k*x*z*forcing` and its selected JVP with respect to the local
unknown `z`. GlassHelix describes the scientific model, quantity roles and
immutable axes with `system_definition`, then reads the result through its
typed observation map with source and value-generation provenance. This is an
engineering fixture; biological learning remains `not_run`.

Run the controlled CPU check:

```sh
python3 -B docs/learning/check_native_bridge.py --fixture \
  --cellerator-dir /home/tumlinson/Cellerator/build-ml2-cpp \
  --work-dir /tmp/gh-ml2-native-fixture \
  --install-prefix /tmp/gh-ml2-installed-sdk
```

After committing the public GH sources, rerun that command to install the same
source used by the installed Python training consumer. The final fitted check
consumes the actual consumer outputs:

```sh
python3 -B docs/learning/check_native_bridge.py \
  --values /path/to/bridge-values.txt \
  --checkpoint /path/to/checkpoint.json \
  --producer-receipt /path/to/receipt.json \
  --cellerator-dir /home/tumlinson/Cellerator/build-ml2-cpp \
  --work-dir /tmp/gh-ml2-native-fitted \
  --install-prefix /tmp/gh-ml2-installed-sdk
```

The checker configures/builds/installs GH with `BUILD_TESTING=OFF`, builds this
external consumer against that install, and executes it. It records source
commits, exact SDK archive hashes, package identity hashes, executable hash,
checkpoint hash, producer receipt hash, values hash and build log. Fitted
acceptance rejects dirty public GH or bridge sources, changed frozen scientific
identities, and a checkpoint hash that disagrees with the supplied values.
The controlled CPU check reports its dirty-source state and does not qualify
installed CT training. Both checks use host arithmetic and require the CUDA
toolkit only because it is a public SDK dependency.

The text input contains one whitespace-separated `key value` per line. Duplicate
keys are rejected by the executable; unknown keys permit checkpoint/source hash
bindings. Required numerical fields are `shared_coefficient`, `input_x`,
`local_z`, `forcing`, `expected_output`, `derivative_z`, `observed_mask`, `scale`,
`tolerance` and `time`. The relative/absolute comparison tolerance must be
positive and at most `1e-3`. Observation masks are exactly `0` or `1`.

Frozen fields are model `7001`, revision `1`, source domain `101`, order `102`, geometry
`103`, partition `104`, structure `105`, structure epoch `1` and evidence `8001`.
Their keys are `model_id`, `model_revision`, `domain_id`, `order_id`,
`geometry_id`, `partition_id`, `structure_id`, `structure_epoch`, `evidence_id`.
The source axis has `source_extent 2`. The separate response axis uses
`target_domain_id 201`, `target_order_id 202`, `target_geometry_id 203`,
`target_partition_id 204`, `target_extent 1`. The shared parameter axis uses
`parameter_domain_id 301`, `parameter_order_id 302`, `parameter_geometry_id 303`
and `parameter_partition_id 304`. The manifest maps
`hypothesis_label synthetic-product-7001-v1` to typed model `7001` and preserves
the producer's `mechanism_id 501`. Each CT axis identity is the full pair
`(low, low + 1000)` in this frozen fixture; the producer identity mapping is
checked before acceptance, including mechanism `(501,1501)` and coefficient
`(401,1401)`. The scalar `x` and `z` are selected values
from the source axis; their roles remain state and local unknown.
The remaining required fields are `mechanism_id`, `state_generation`,
`parameter_generation`, `units`, `modality`, `time_units` and `sampling_id`.
Fitted input additionally supplies `checkpoint_sha256`. Preserve the actual
checkpoint values and generations; changing a generation does not change an
immutable axis or structure epoch.
