`native_cluster_restart.txt` contains the host `<GEN>` and cluster SCF
summaries from a serial SPR-KKR 9.7 continuation of the user's Fe3CoFe3
cluster on 2026-11-18. The input/output potentials are represented by the
fixtures in `potentials/examples`.

The intervening lattice/core-density diagnostics were omitted and a minimal
FILES header was retained. The numerical summaries are unchanged. The host
summary includes an optional `P(T_z)` / `T_z` table and has no SCF iteration
footer; only the later cluster summary completes iteration 1. This catches
both the optional-table parse failure and accidental assignment of the host
atomic types to the cluster iteration.
