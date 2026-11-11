#!/usr/bin/env python3
"""Generate the training USD asset using Isaac Lab 3.0's MJCF converter.

Run only inside the Isaac Lab 3.0 training environment, after the MJCF compiles.
The importer wheel may need to be installed in minimal/kit-less setups.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    root = args.repo.expanduser().resolve()
    path = root / 'robot_model/a2_piper_x.xml'
    dest = root / 'robot_model/generated_usd/a2_piper_x'
    if not path.is_file():
        ap.error(f'Missing MJCF: {path}')
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    cfg = MjcfConverterCfg(
        asset_path=str(path),
        usd_dir=str(dest),
        usd_file_name='a2_piper_x.usda',
        force_usd_conversion=args.force,
        merge_mesh=False,
        collision_from_visuals=False,
        self_collision=False,
        fix_base=False,
    )
    output = MjcfConverter(cfg)
    print('Generated asset:', output.usd_path)
    if not Path(output.usd_path).is_file():
        raise RuntimeError(f'Importer did not write the expected USD: {output.usd_path}')
    print('USD conversion: PASS (structural and dynamic validation still required)')


if __name__ == '__main__':
    main()
