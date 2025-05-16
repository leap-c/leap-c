import textwrap
from dataclasses import dataclass, field
from leap_c.utils.cfg import cfg_as_python


@dataclass
class VeryDeep:
    z: float = 1.5


@dataclass
class SomeClass:
    x: int = 3
    y: tuple[int, ...] = (12, 13)
    deep: VeryDeep = field(default_factory=VeryDeep)


@dataclass
class OtherClass:
    name: str = "abc"
    blob: int = 3


@dataclass
class DummyParent:
    some: SomeClass = field(default_factory=SomeClass)
    other: OtherClass = field(default_factory=OtherClass)
    param: int = 3


def test_print_cfg() -> None:
    expected_output: str = textwrap.dedent("""\
        # ---- Configuration ----
        cfg = DummyParent()
        cfg.param = 3

        # ---- Section: cfg.some ----
        cfg.some.x = 3
        cfg.some.y = (12, 13)

        # ---- Section: cfg.some.deep ----
        cfg.some.deep.z = 1.5

        # ---- Section: cfg.other ----
        cfg.other.name = 'abc'
        cfg.other.blob = 3
    """).strip()

    output: str = cfg_as_python(DummyParent(), root_name="cfg")
    assert output.strip() == expected_output
