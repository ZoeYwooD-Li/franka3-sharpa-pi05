"""Replay a saved IK failure offline, without importing any hardware path."""

import argparse
import json
from pathlib import Path

import numpy as np

from .actions import TrexActionDecoder
from .contract import URDF_PATH
from .kinematics import ArmKinematics, IKError


def replay(record, urdf=URDF_PATH):
    saved = record.get('solver')
    if not saved:
        raise ValueError('Record has no solver residual diagnostics')
    model = ArmKinematics(urdf, saved['side'])
    if model.urdf_sha256 != saved['urdf_sha256']:
        raise ValueError('Replay URDF SHA256 does not match the failed solve')
    try:
        model.inverse(saved['target'], saved['seed'],
                      position_tolerance=saved['position_tolerance_m'],
                      orientation_tolerance=saved['orientation_tolerance_rad'],
                      max_iterations=saved['max_iterations'])
    except IKError as error:
        current = error.diagnostics
        if current is None:
            raise ValueError('Replay stopped before residual diagnostics') from error
    else:
        return {'reproduced': False, 'outcome': 'solver_accepted'}

    solver_matches = (
        current['stop_reason'] == saved['stop_reason']
        and current['iterations'] == saved['iterations']
        and np.allclose(current['final_q'], saved['final_q'], rtol=0, atol=1e-12)
        and np.isclose(current['position_error_m'], saved['position_error_m'], rtol=0, atol=1e-12)
        and np.isclose(current['orientation_error_rad'], saved['orientation_error_rad'], rtol=0, atol=1e-12)
    )
    # Also reconstruct the fixed chunk base and preceding row seeds from the
    # untouched model output, independently of the saved target/seed pair.
    try:
        TrexActionDecoder(urdf).decode(record['raw_actions'], record['measured_state'])
    except IKError as error:
        decoded = error.diagnostics
        chunk_matches = bool(decoded and decoded['side'] == saved['side']
                             and decoded['action_index'] == saved['action_index']
                             and np.allclose(decoded['target'], saved['target'], rtol=0, atol=1e-12)
                             and np.allclose(decoded['seed'], saved['seed'], rtol=0, atol=1e-12))
    else:
        chunk_matches = False
    return {'reproduced': bool(solver_matches and chunk_matches),
            'solver_matches': bool(solver_matches), 'chunk_matches': chunk_matches,
            'side': saved['side'], 'action_index': saved['action_index'],
            'stop_reason': current['stop_reason'], 'iterations': current['iterations'],
            'position_error_m': current['position_error_m'],
            'orientation_error_rad': current['orientation_error_rad']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, help='Saved ik-failure.json')
    parser.add_argument('--urdf', type=Path, default=URDF_PATH)
    args = parser.parse_args(argv)
    result = replay(json.loads(args.record.read_text()), args.urdf)
    print(json.dumps(result, indent=2))
    return 0 if result['reproduced'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
