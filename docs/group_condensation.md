# Response-preserving group condensation

`parastell.group_condensation` provides a deterministic adjacent-group merge
algorithm for synthetic validation and later nuclear-data response studies.

Inputs are fine-grid boundaries, one or more integrated spectra, protected
fine-group response coefficients, mandatory boundaries, a relative response
tolerance, and group-count bounds. The algorithm:

1. preserves every mandatory boundary exactly;
2. sums spectrum/current weights exactly into coarse groups;
3. evaluates every supplied spectrum-response pair after each candidate merge;
4. accepts the deterministic lowest-error merge within tolerance;
5. records every removed boundary and resulting error;
6. refuses to register a result unless both tolerance and maximum-group criteria
   pass.

The included helper boundary lists cover thermal, resonance, D-D, broad fission,
D-T, 14.1 MeV, and 20 MeV regions for neutrons, plus the pair-production threshold
for photons. A production study must insert exact material reaction thresholds,
absorption edges, and prompt/decay lines into its fine master grid before calling
the condenser.

This checkpoint validates the algorithm with synthetic responses. It does not
qualify `response-selected-neutron-v1` or `response-selected-photon-v1`; those names
require real heating, activation, gas-production, damage, and PKA response studies.
