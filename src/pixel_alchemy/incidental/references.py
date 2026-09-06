import pathlib

REF_DIR = pathlib.Path(__file__).with_name("reference")

# canonical reference set
REFERENCE_FILES = {
    "ambient_darin": "ambient_darin.rb",
    "chord_cycle": "chord_cycle.rb",
    "ocean": "ocean.rb",
    "ambient_sampling": "ambient_sampling.rb",
    "haunted": "haunted.rb",
    "choral": "choral.rb",
}


def list_references() -> list[str]:
    return sorted(REFERENCE_FILES.keys())


def load_reference(name: str) -> str:
    fn = REFERENCE_FILES.get(name)
    if fn is None:
        raise KeyError(f"unknown reference {name!r}, known: {list(REFERENCE_FILES)}")
    return (REF_DIR / fn).read_text()


def reference_path(name: str) -> pathlib.Path:
    fn = REFERENCE_FILES.get(name)
    if fn is None:
        raise KeyError(f"unknown reference {name!r}")
    return REF_DIR / fn
