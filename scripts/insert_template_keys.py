# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Insert config-template entries for Pydantic-aliased fields.

Generates Type/Default/Source/Description blocks (the
``config_template_comprehensive.yaml`` house style) straight from a Pydantic
model's fields, and inserts them after an anchor line. Used when promoting
flat keys into typed config models so the config-authority test
(every core alias documented in the comprehensive template) stays green.

Usage:
    python scripts/insert_template_keys.py <dotted.module:Class> \
        <ALIAS1,ALIAS2,...> <anchor-line-prefix> [template-path]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATE = (
    "src/symfluence/resources/config_templates/config_template_comprehensive.yaml"
)


def _yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _type_name(annotation: Any) -> str:
    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")


def render_blocks(model_cls: type, aliases: list[str], source_label: str) -> str:
    by_alias = {f.alias: (n, f) for n, f in model_cls.model_fields.items() if f.alias}
    blocks = []
    for alias in aliases:
        if alias not in by_alias:
            raise SystemExit(f"alias {alias} not found on {model_cls.__name__}")
        _, field = by_alias[alias]
        default = field.default
        if default is None and field.default_factory is not None:
            default = field.default_factory()
        description = field.description or alias.replace("_", " ").title()
        blocks.append(
            f"# {alias}\n"
            f"#   Type:        {_type_name(field.annotation)}\n"
            f"#   Default:     {default}\n"
            f"#   Source:      {source_label}\n"
            f"#   Description: {description}\n"
            f"{alias}: {_yaml_value(default)}\n"
        )
    return "\n" + "\n".join(blocks)


def main() -> None:
    module_path, cls_name = sys.argv[1].split(":")
    aliases = sys.argv[2].split(",")
    anchor = sys.argv[3]
    template = Path(sys.argv[4] if len(sys.argv) > 4 else DEFAULT_TEMPLATE)

    model_cls = getattr(importlib.import_module(module_path), cls_name)
    text = template.read_text()
    lines = text.splitlines(keepends=True)
    idx = next(
        i for i, line in enumerate(lines) if line.startswith(anchor)
    )
    insertion = render_blocks(
        model_cls, aliases, f"{cls_name} ({module_path.rsplit('.', 1)[-1]}.py)"
    )
    lines.insert(idx + 1, insertion)
    template.write_text("".join(lines))
    print(f"inserted {len(aliases)} entries after '{anchor}' in {template}")


if __name__ == "__main__":
    main()
