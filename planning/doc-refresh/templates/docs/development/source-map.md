# Source map: scientific objects to executable tests

## Scientific objects and supplied alternatives

- **Which scientific objects are distinguished?** Start at [core contracts](../../{{P_CORE}}) for quantities, observations, identity/provenance, and state roles.
- **How are supplied alternatives updated?** Read [finite candidates](../../{{P_INFERENCE}}) for whole joint state–mechanism alternatives and evidence history.
- **Where does a system become executable?** See [assembly](../../{{P_ASSEMBLY}}), which delegates numerical execution to Cellerator.

## Build and evidence paths

- **Where are supported capabilities checked?** Start at [native foundation tests](../../{{P_TESTS}}) for contract and numerical/semantic checks.
- **Where is the complete-program measurement?** The [benchmark source](../../{{P_BENCH}}) and [recorded methodology](../../{{P_BENCH_REPORT}}) describe bounded lifecycle checks and the limits of activity semantics.
- **How are replay and Python connected?** Follow [replay](../../{{P_REPLAY}}), the [binding](../../{{P_BINDINGS}}), and the [Python layer](../../{{P_PYTHON}}). These are the configured paths, not a commitment to a universal package layout.

The [build graph](../../{{P_CORE_CMAKE}}) defines `glasshelix_foundation` from `src/system/assembly.cc`, linking Cellerator's `native_foundation` and `indexed_mechanism` components through the core interface. `BUILD_TESTING` enables CUDA and native checks; replay and Python targets are conditional on their source files. See the [build guide](start.md) for the minimal library command. No source move was made: these semantic boundaries already provide a useful map, while legacy material needs a separate consumer review before any relocation.
