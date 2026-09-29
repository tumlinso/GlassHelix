# GlassHelix evidence selection

No candidate qualifies for public quantitative display in this checkout. The current [results index](../../docs/results/index.md) explains this boundary. Selection is based on evidence availability and explanatory value, not speed.

| Candidate | Disposition | Evidence state | Reason / retained claim boundary |
|---|---|---|---|
| GH-ACCEPT complete-program lifecycle | Omitted from public results; keep the local prepared record only | Recorded summary, not verified historical | The report identifies measurement source commit `1c22d320fd65b21976d159792e2e93b2c4af7120`; its benchmark source is present and has the same SHA-256 at that commit and current HEAD. However the referenced raw CSV, lease receipt and `/tmp/nf1a-gh-benchmark-result.json` are missing. The claimed CSV SHA-256 cannot be recomputed; Project Control has no matching CUDA evidence for controller ID `824295c5-ff13-4ff6-96fb-3822a02788dc`. The summary values and ranges therefore remain unauthenticated and are not displayed. No speedup claim is supported. |
| Ambiguity/perturbation capability fixture | Omitted; not run | Pending | The fixture and analytical oracle are prepared, but no call to the actual GlassHelix library has been executed. The oracle is not product evidence or biological validation. |

No raw samples, lease details, or confidence intervals were reconstructed. The historical report remains unchanged. A future reviewer can reconsider GH-ACCEPT if the original CSV and lease/result artifacts become available and their identities match the recorded SHA-256 and source.
