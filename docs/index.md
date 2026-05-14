# Welcome to Bloqade-Core

Bloqade-Core provides the core building blocks shared across the Bloqade
ecosystem. It currently exposes the `bloqade.geometry` namespace, a collection
of tools for transforming and modeling geometric objects used for defining
layouts and operation of neutral atom quantum computers.

Currently the only supported geometry is a grid, but more geometries will be added in
the future. For a full list of features, see the [API Reference](reference/bloqade/geometry/prelude/).

## Installation

```bash
uv add bloqade-core
```

See [Installation](install.md) for more details.

## Other useful links

- [Bloqade Shuttle](https://queracomputing.github.io/bloqade-shuttle/dev/): a related
project that uses `bloqade.geometry` to define and manipulate atom shuttling operations
in neutral atom quantum computers.
