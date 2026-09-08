# Lane and milestone guide

This project contains 16 first-class lanes, including the coordinator. Each queue below is ordered. The combined graph is in `machine/joint_topology.json`.

| Lane | Role | Tasks |
|---|---|---|
| GH-NF1-L-A | implementer | A01, A02, A03, A04 |
| GH-NF1-L-C | implementer | C01, C02, C03, C04, C05, C06, C07 |
| GH-NF1-L-B | implementer | B01, B02, B03, B04, B05 |
| GH-NF1-L-S | implementer | S01, S02, S03, S04, S05, S06, S07 |
| GH-NF1-L-D | implementer | D01, D02, D03, D04, D05, D06, D07 |
| GH-NF1-L-O | implementer | O01, O02, O03, O04, O05, O06 |
| GH-NF1-L-E | implementer | E01, E02, E03, E04, E05 |
| GH-NF1-L-Q | implementer | Q01, Q02, Q03, Q04, Q05, Q06, Q07 |
| GH-NF1-L-R | implementer | R01, R02, R03, R04, R05 |
| GH-NF1-L-P | implementer | P01, P02, P03, P04, P05, P06 |
| GH-NF1-L-U | implementer | U01, U02, U03, U04, U05, U06 |
| GH-NF1-L-T | validator | T01, T02, T03, T04, T05, T06 |
| GH-NF1-L-K | specialist | K01, K02, K03, K04, K05 |
| GH-NF1-L-X | integrator | X01, X02, X03, X04, X05, X06 |
| GH-NF1-L-M | integrator | M00, M10, M20, M30, M40, M60, M90 |

## Integration does not wait behind its own build hooks

The first baseline integration may seed a small optional build-fragment hook before feature fan-out, with no placeholder capabilities. Provider/build lanes then register real tests through their assigned fragments. If an existing build requires a central change not yet claimed, the controller records a narrow prerequisite integration/scope transfer; it must not wait until a milestone whose prerequisites need those tests. This is an explicitly authorized coordination adjustment, not an excuse to accept zero tests.

Only actual accepted prerequisite work releases the next milestone. Contract receipt can release design and interface work; linked/GPU consumer claims require the later capability receipts.
