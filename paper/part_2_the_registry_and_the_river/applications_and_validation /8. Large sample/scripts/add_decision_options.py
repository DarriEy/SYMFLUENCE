#!/usr/bin/env python3
"""Add restricted FUSE_DECISION_OPTIONS to all large-sample configs.

Based on Section 4.6 findings, only QPERC and QINTF are statistically
significant decisions. All others are fixed to their established values.
This reduces the decision ensemble from 3456 to 4 combinations per domain.
"""

from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Decision options block: only vary the two key decisions from Section 4.6
# All other decisions fixed to values used/identified in earlier sections
DECISION_BLOCK = """
# FUSE decision options — restricted to key decisions from Section 4.6
# Only QPERC and QINTF varied (2x2 = 4 combinations); all others fixed.
FUSE_DECISION_OPTIONS:
  RFERR: [multiplc_e]
  ARCH1: [tension1_1]
  ARCH2: [tens2pll_2]
  QSURF: [arno_x_vic]
  QPERC: [perc_f2sat, perc_lower]
  ESOIL: [sequential]
  QINTF: [intflwnone, intflwsome]
  Q_TDH: [rout_gamma]
  SNOWM: [temp_index]
"""

ANCHOR = "# DDS specific settings"


def main():
    configs = sorted(CONFIGS_DIR.glob("config_lamahice_*_FUSE.yaml"))
    print(f"Found {len(configs)} configs")

    updated = 0
    for cfg in configs:
        text = cfg.read_text()

        if "FUSE_DECISION_OPTIONS" in text:
            print(f"  SKIP (already present): {cfg.name}")
            continue

        # Insert before the DDS specific settings line (end of optimization section)
        if ANCHOR in text:
            text = text.replace(ANCHOR, DECISION_BLOCK + ANCHOR)
        else:
            # Fallback: append to end of file
            text = text.rstrip() + "\n" + DECISION_BLOCK

        cfg.write_text(text)
        updated += 1

    print(f"\nUpdated {updated} configs with FUSE_DECISION_OPTIONS (4 combinations)")


if __name__ == "__main__":
    main()
