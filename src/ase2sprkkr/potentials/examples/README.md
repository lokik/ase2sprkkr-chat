`Fe3CoFe3.clu.iterating.pot` is the unmodified native CLUSCF output from
`~/tmp/Project/Fe3CoFe3.clu.pot_new`, supplied for a reader regression on
2026-11-18. `Fe3CoFe3.host.pot` is the corresponding unmodified periodic host
`Fe3CoFe3.pot` from the same directory.

The output has seven cluster sites, three local types, reused host meshes
(`NMCLU=0`), global radial TYPE identifiers 3 through 5, padded quoted SCF and
lattice values, and the two-component `NE 32 0` record. Its native status is
`ITERATING`, with SCF iteration 34. Tests load it directly and reconstruct
atoms and radial data against the host without running SPR-KKR.
