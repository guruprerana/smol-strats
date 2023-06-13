# smol-strats

Synthesizing compact strategies for MDPs specified in the PRISM syntax

## Dependencies

1. Storm, CaRL, pycarl, stormpy

## Important commands

1. `storm --prism benchmarks/generated/generated.prism --prop "Pmax=? [ F \"target\" ]"`
