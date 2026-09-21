# NF1 replay and Python access

`GlassHelix::nf1_replay` records one defined NF1 logical operation, currently
`local-add#10@1`. A record carries the registered definition identity, the
`f32-rne-propagate` policy, logical JVP inputs and output, and the exact
40-hex GlassHelix and Cellerator build identities. The replay CLI receives the
configured/runtime GH and CE identities explicitly and refuses a mismatch
before it calls the Cellerator operation.

The record format is deliberately narrow. It rejects control characters,
separators in atoms, trailing numeric commas, nonfinite values, unknown schema,
unknown policy, and invalid status/output combinations. It represents pending,
failed, successful-empty, and successful-with-output separately. Replay only
accepts successful output records for the registered operation; a different
registered block is preserved as metadata but is not silently interpreted.

The installed native consumer links `GlassHelix::nf1_replay` and invokes
`nf1_replay_cli artifact GH_SOURCE CE_SOURCE`. The Python consumer imports
`glasshelix_nf1` and exposes host JVP for contiguous one-dimensional float32
buffers with no implicit copies. It retains those buffer owners while the GIL is
released. VJP and GPU interchange report unsupported capabilities.

The ANALYZE CMake integration must install target `GlassHelix::nf1_replay` and
module target `glasshelix_nf1`; the access test script is the corresponding
fresh consumer check.
