import textwrap
from dataclasses import dataclass, field
from leap_c.print_cfg import print_cfg_as_python


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
        cfg = DummyParent()
        cfg.some.x = 3
        cfg.some.y = (12, 13)
        cfg.some.deep.z = 1.5
        cfg.other.name = 'abc'
        cfg.other.blob = 3
        cfg.param = 3
    """).strip()

    output: str = print_cfg_as_python(DummyParent(), root_name="cfg")
    assert output.strip() == expected_output
