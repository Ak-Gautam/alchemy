# Rust Extensions (Future)

This directory is reserved for Rust performance extensions via PyO3 + maturin.

Planned components:
- **Fast structural validation** — validate thousands of samples against a schema
- **Deduplication** — similarity hashing across large sample sets
- **Batch serialization** — high-throughput output writing

The Rust extension will be importable as `alchemy._core` with a Python fallback.
