# Standard parts: bd_warehouse

`bd_warehouse` is a Build123d-native companion library bundled as a
default dependency. Reach for it before hand-rolling any standard
hardware. Its classes subclass build123d's `BasePartObject`, so an
instance *is* a `Part` and drops straight into an `Assembly`.

## What's in it

The submodules at the time of writing — discover, don't trust this
list:

- `fastener` — screws, nuts, washers (ISO/DIN/imperial), tapped /
  clearance / countersunk holes
- `bearing` — deep-groove / angular / cylindrical / tapered ball+roller
  bearings, plus `PressFitHole`
- `thread` — `IsoThread`, `AcmeThread`, `MetricTrapezoidalThread` for
  modeled (turnable) threads
- `gear` — `InvoluteToothProfile`, `SpurGear`
- `sprocket`, `pipe`, `flange` — chain sprockets, NPS pipes, raised-face
  flanges
- `open_builds` — V-slot / C-beam extrusions, gantry plates, eccentric
  spacers, lead screws, flexible couplers (OpenBuilds CNC parts)

Submodules are not auto-imported at the top level — `import bd_warehouse`
alone exposes nothing useful. Always import the submodule you want.

## Discovering what's available

Three layers of introspection, cheapest first.

### 1. `inspect.signature(Class)` — parameters and valid type codes

This is the highest-signal channel. Many `bd_warehouse` classes annotate
their type-selector arguments with `Literal[...]`, so the signature
*enumerates* the valid codes:

```python
import inspect
from bd_warehouse.fastener import HexNut

print(inspect.signature(HexNut))
# (size: 'str',
#  fastener_type: "Literal['iso4032', 'iso4033', 'iso4035']" = 'iso4032',
#  hand: "Literal['right', 'left']" = 'right',
#  simple: 'bool' = True, ...)
```

Use this first whenever the question is "what fastener_type / bearing_type
strings does this class accept?".

### 2. `dir(submodule)` — class names in a submodule

```python
import bd_warehouse.fastener as f
classes = [n for n in dir(f)
           if not n.startswith("_") and isinstance(getattr(f, n), type)]
```

Filter by `isinstance(getattr(f, n), type)` — `dir()` also surfaces
re-exports from build123d (`Axis`, `BuildPart`, `Compound`, etc.) that
aren't part of the warehouse API.

### 3. Source files — when introspection is awkward

For "every class anywhere in the package" or full source reading:

```bash
ls .venv/lib/python*/site-packages/bd_warehouse/
grep -nE "^class " .venv/lib/python*/site-packages/bd_warehouse/fastener.py
```

The [bd_warehouse docs](https://bd-warehouse.readthedocs.io/en/latest/)
are the better source for *usage examples and idioms*; introspection
and source are the better source for *does this class exist and what
arguments does it take*.

## Size strings: read the error

Sizes are specific — `"M8-1.25"` is M8 × 1.25 mm pitch. The valid set
isn't enumerated in the signature (it's data-driven from a table), so
discover it by triggering the validation:

```python
HexNut(size="bogus", fastener_type="iso4032")
# ValueError: Invalid size 'bogus'. Valid sizes are: ['M1.6-0.35', 'M2-0.4', ...]
```

Read the error, pick from the list. This works for every class with a
`size=` parameter.

## Usage pattern

Wrap each `bd_warehouse` call in a thin pure part function so the
script stays readable and the size lives with the rest of the design
parameters:

```python
from build123d import Location, Part
from bd_warehouse.fastener import HexNut
from cad_khana.mechanism.assembly import Assembly

NUT_SIZE = "M8-1.25"

def lock_nut(size: str = NUT_SIZE) -> Part:
    return HexNut(size=size, fastener_type="iso4032")

assembly = Assembly().with_part("nut", lock_nut(), location=Location((0, 0, 10)))
```

## A screw's threaded length is drawn at the minor Ø

`simple=True` (the default) leaves the thread out of both parts: a
screw's threaded length stops at the thread's **minor** Ø (Ø2.459 for
an M3), and a nut's bore at the **major** Ø, so a simple screw and nut
stand 0.27 mm apart all round, the depth the threads would mesh across,
and a fused screw, nut and washer stack comes out as several solids.
`simple=False` draws the thread, but as separate solids (34 for an
M3×16) that make distance and overlap checks 80 to 180 times slower,
and a threaded screw and nut fuse into an invalid solid.

For a part a claim measures — a hole's clearance, a fit, a stack fused
into one part — keep `simple=True` and fuse on the thread's outer
envelope, a major-Ø cylinder along the threaded length:

```python
from build123d import Cylinder, Part, Pos
from bd_warehouse.fastener import SocketHeadCapScrew

def m3_screw(length: float = 16) -> Part:
    screw = SocketHeadCapScrew("M3-0.5", length, "iso4762")
    return screw + Pos(0, 0, -length / 2) * Cylinder(3.0 / 2, length)
```

Leave it drawn as is where the minor Ø is the point, such as a screw in
a tap-drill hole.

## Don't inspect bought parts

Skip `inspect()` for anything from `bd_warehouse` — fasteners, bearings,
extrusions, and the rest are bought, not printed. Printability checks
on standard hardware are noise.
