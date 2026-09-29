# GlassHelix

**What can we still learn about a cell when several hidden systems could explain the same measurements?**

A useful model can predict what happens next without uniquely identifying why. GlassHelix investigates the information that remains available in that gap: responses shared by competing explanations, state variables needed to account for history, local sensitivities, and observations or interventions that could tell alternatives apart.

The aim is not to make complete reconstruction a prerequisite for insight. It is to build methods that retain defensible pieces of biological dynamics and mechanism while making the limits of the evidence explicit.

## From observations to testable alternatives

GlassHelix separates measurements, dynamical state and proposed mechanism. It can compare whole alternatives without declaring that a predictive latent coordinate is a molecule or that a population snapshot is an individual trajectory. An unresolved family of explanations can be a useful result, not a failure to choose a model.

Its scientific formulation remains expandable: no single neural architecture, state definition or dynamical law is assumed to describe every biological system. Concrete implementations make narrower choices that must be stated and tested.

A longer-term goal is for scientific learning to change the executable representation itself. A supported state variable or dependency could become explicit structure for Cellerator, reducing repeated reconstruction where evidence permits. An unknown current value may still require measurement or state estimation. Automatic promotion of mechanisms is an ambition, not a delivered general feature.

## What exists now

{{README_STATUS}}

## See the evidence

{{README_RESULTS}}

The [results pages](docs/results/index.md) distinguish synthetic execution/capability checks from fitted biological results. They show what was compared, what remained ambiguous and which parts of the computation were actually exercised.

## Find your way in

Read the [scientific/design overview](docs/design/overview.md), then the [source map](docs/development/source-map.md) or [build/use guide](docs/development/start.md). The [current snapshot](docs/status/current.md) names the supported foundation and learning work still under development. Older model sketches are kept separate from that current code.

[How the projects fit together](docs/design/program.md): GlassHelix asks the scientific question; [Cellerator](https://github.com/tumlinso/Cellerator) supplies structured numerical execution; [Baseplane](https://github.com/tumlinso/Baseplane) develops the route toward sequence-grounded representation.

This is research software. Its existing toolbox does not settle the general identifiability or mechanism-discovery problem.
