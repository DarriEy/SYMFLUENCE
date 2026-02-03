#!/usr/bin/env python3
"""Generate FUSE configuration files for all LamaH-Ice catchments.

Reads a base FUSE config template and generates per-catchment YAML configs
with the correct domain names and paths for the Large Sample study.
"""

import argparse
from pathlib import Path


# LamaH-Ice domain IDs for the large sample study
# Note: For full setup with time period detection, use setup_all_domains.py instead.
# This script uses simple template substitution (same time periods for all domains).
DOMAIN_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
    890, 990, 1010, 9900,
]

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
TEMPLATE_CONFIG = CONFIGS_DIR / "config_lamahice_42_FUSE.yaml"


def generate_config(domain_id: int, template_path: Path, output_dir: Path) -> Path:
    """Generate a FUSE config for a single LamaH-Ice domain."""
    output_path = output_dir / f"config_lamahice_{domain_id}_FUSE.yaml"

    content = template_path.read_text()
    content = content.replace("Domain 42", f"Domain {domain_id}")
    content = content.replace("DOMAIN_NAME: lamahice_42", f"DOMAIN_NAME: lamahice_{domain_id}")
    content = content.replace("LAMAH_ICE_DOMAIN_ID: 42", f"LAMAH_ICE_DOMAIN_ID: {domain_id}")

    output_path.write_text(content)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate FUSE configs for LamaH-Ice catchments")
    parser.add_argument("--template", type=Path, default=TEMPLATE_CONFIG,
                        help="Path to template config YAML")
    parser.add_argument("--output-dir", type=Path, default=CONFIGS_DIR,
                        help="Output directory for generated configs")
    parser.add_argument("--domains", nargs="+", type=int, default=DOMAIN_IDS,
                        help="Domain IDs to generate configs for")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.template.exists():
        raise FileNotFoundError(f"Template config not found: {args.template}")

    print(f"Template: {args.template}")
    print(f"Output:   {args.output_dir}")
    print(f"Domains:  {args.domains}")
    print()

    for domain_id in args.domains:
        path = generate_config(domain_id, args.template, args.output_dir)
        print(f"  Generated: {path.name}")

    print(f"\nDone. Generated {len(args.domains)} config files.")


if __name__ == "__main__":
    main()
