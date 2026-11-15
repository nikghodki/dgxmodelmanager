"""Final review I5: the documented nodes example is valid configuration, and the node field table matches NodeConfig."""
import re
from pathlib import Path

import yaml

from dgxmm.config import NcclConfig, NodeConfig

DOC = Path(__file__).resolve().parent.parent / "docs" / "configuration" / "index.md"


def _nodes_section() -> str:
    text = DOC.read_text()
    start = text.index("### Nodes")
    end = text.index("\n### ", start + 1)
    return text[start:end]


def test_documented_nodes_example_loads_through_settings(settings_factory):
    blocks = re.findall(r"```yaml\n(.*?)```", _nodes_section(), re.S)
    example = next(yaml.safe_load(b) for b in blocks if b.startswith("nodes:"))
    s = settings_factory(**example)
    assert [n.id for n in s.all_nodes()][0] == "local"
    assert all(n.docker_url.startswith("ssh://") for n in s.all_nodes()[1:])


def test_node_field_table_lists_exactly_the_nodeconfig_fields():
    rows = re.findall(r"^\| `([a-z_.]+)` \|", _nodes_section(), re.M)
    documented = {r for r in rows}
    fields = set(NodeConfig.model_fields) - {"nccl"} | {f"nccl.{f}" for f in NcclConfig.model_fields}
    assert documented == fields
