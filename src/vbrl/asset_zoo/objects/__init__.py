from pathlib import Path

OBJECTS_DIR = Path(__file__).resolve().parent

CUBE_XML = OBJECTS_DIR / "cube.xml"
PUSH_T_XML = OBJECTS_DIR / "push_t.xml"


def push_chain_xml(num_links: int) -> Path:
  return OBJECTS_DIR / f"push_chain_{num_links}.xml"


__all__ = [
  "CUBE_XML",
  "OBJECTS_DIR",
  "PUSH_T_XML",
  "push_chain_xml",
]
