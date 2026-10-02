# Installed native capability receipt

`ce-native-acceptance.json` bridges the completed Cellerator product2 capability to GlassHelix. The new isolated `/tmp/gh-moon-celleratorch` installation contains a byte-identical copy of the final CE Python package. `install-manifest.json` records the install command, source commit, package hashes, interpreter and separately installed native library hash.

`installed_consumer.py` exercises the installed public `product2`, `product2_jvp` and `Product2Module` calls on CPU FP32. Its four-product fixture includes a repeated input and zero inputs. An independent scalar reference checks forward, input and coefficient VJPs and the full JVP. Finite differences and the adjoint identity cross-check the derivatives; an external SGD step consumes the module coefficient gradient. The result and command logs are retained beside the consumer.

Producer CPU/native CUDA correctness evidence is reused from CE's frozen source and controller receipts. This receipt qualifies the Torch CPU FP32 route. CUDA tensors in the Torch adapter, mixed precision, second-order transforms, performance and biological inference remain outside the claim. Retained external installed artifacts are required for verification.

Run the consumer with the exact command recorded in the receipt. Run `python3 experiments/moonshot-parallel-v1/receipt/verify_consumer_evidence.py` for a pure hash/provenance check. The planning `check_receipt.py` separately verifies the CE task's effective completion and root acceptance. The root controller reviewed and accepted the consumer and its source-bound evidence; the receipt records that acceptance time.
