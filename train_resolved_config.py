#!/usr/bin/env python3
"""Train from one fully resolved ARC-BPO YAML config.

This entry point is used by controlled experiment launchers. It deliberately
does not compose Hydra defaults: every scientific and operational setting must
already be present in the supplied config saved by the launcher.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Fully resolved training YAML.")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    # Imported lazily because train.py intentionally depends on Unix `resource`
    # and is meant to run on the Linux CUDA training server.
    import train

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Resolved training config has missing keys: {sorted(missing)}")
    train.main.__wrapped__(config)


if __name__ == "__main__":
    main()
