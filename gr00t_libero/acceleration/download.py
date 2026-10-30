"""Download the pinned LIBERO-10 W1.5A8 fixed-input reproduction bundle."""
import argparse
import json
from pathlib import Path

from ..download_inference import materialize
from .prepare import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'presets/inference_libero10_w1p5a8.json')
    parser.add_argument('--local-model', type=Path)
    parser.add_argument('--local-processor', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['inference']['schema'] != 'sacq_a8_fixed_input_bundle_v1':
        raise ValueError('Expected the A8 bundle manifest')
    materialize(manifest, args.output, local_model=args.local_model,
                local_processor=args.local_processor)


if __name__ == '__main__':
    main()
