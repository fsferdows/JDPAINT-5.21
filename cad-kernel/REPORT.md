## `jdpaint_rust_core` in Rust

![jdpaint_rust_core report card](card.svg)

| | |
| --- | --- |
| Unit | `jdpaint_rust_core::vectorize + poisson` (Python / Rust) |
| Recorded cases matched | 6 / 6 |
| Speedup, call included | 4.3x (geometric mean) |
| Fallback to the original | 0.0% (0 of 6 cases) |
| Switched on | on |

| Function | Cases | Original per call | Rust per call | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `vectorize` (Marching Squares + Schneider Cubic) | 4 | 21.5 ms | 2.2 ms | 9.9x |
| `poisson` (4-Level Multigrid Screened Poisson) | 2 | 17.0 ms | 9.0 ms | 1.9x |

Measured following `erkamyaman/rust-anything` `report` schema (`REPORT.md` + `card.svg`) on Apple M1, 8 cores, macOS (note: `rust-anything@5.0.1` CLI itself requires `rustc 1.97.0` while local `rustc` is `1.92.0`, so parity diff + persistent IPC wall-clock timing were executed directly via `cargo test` + live bench).
