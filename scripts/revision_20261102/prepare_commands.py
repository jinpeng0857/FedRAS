"""Materialize opt-in experiment commands; never starts GPU evaluations."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'configs/revision_20261102')
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', required=True, help='Explicit arm IDs; conditional and blocked arms are never silently included.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    env = json.loads((args.config_dir / '00_environment.json').read_text(encoding='utf-8'))
    arms = {}
    for path in sorted(args.config_dir.glob('*.json')):
        for arm in json.loads(path.read_text(encoding='utf-8')).get('arms', []):
            if arm['id'] in arms:
                raise ValueError(f"duplicate arm: {arm['id']}")
            arms[arm['id']] = arm
    unknown = set(args.arms) - arms.keys()
    if unknown:
        raise ValueError(f'unknown arms: {sorted(unknown)}')
    runtime = args.runtime_root.resolve()
    entry = runtime / env['entrypoint']
    if not entry.is_file():
        raise FileNotFoundError(entry)
    output_root = args.output_root.resolve()
    assets = args.asset_root.resolve()
    commands = []
    for arm_id in args.arms:
        arm = arms[arm_id]
        if arm['status'] != 'configured':
            raise ValueError(f"{arm_id} blocked: {arm.get('blockers')}")
        for run_id in arm['checkpoints']:
            anchor = env['anchors'][run_id]
            model = assets / anchor['model_path']
            stats = assets / anchor['eata_statistics_path' if arm.get('method') == 'eata' else 'statistics_path']
            for path in (model, stats, assets / env['corruption_path'], assets / env['clean_path']):
                if not path.exists():
                    raise FileNotFoundError(path)
            case = output_root / arm_id / run_id
            if case.exists():
                raise FileExistsError(f'refusing to reuse existing case directory: {case}')
            cli = dict(env['evaluation_cli'])
            if arm.get('fedras', False):
                cli.update(env['fedras_cli'])
            cli.update(arm['cli'])
            cli.update({'--model_path': str(model), '--statistics_path': str(stats),
                        '--cifar_c_path': str(assets / env['corruption_path']),
                        '--clean_data_root': str(assets / env['clean_path']),
                        '--device': args.device, '--output_path': str(case / 'result.json'),
                        '--review_run_id': f'{arm_id}_{run_id}',
                        '--review_diagnostics_path': str(case / 'diagnostics.jsonl')})
            argv = [args.python, str(entry)]
            for flag, value in cli.items():
                if value is False:
                    continue
                if value is None:
                    raise ValueError(f'{arm_id}: unresolved {flag}')
                argv.append(flag)
                if value is not True:
                    argv.extend(str(x) for x in (value if isinstance(value, list) else [value]))
            commands.append({'arm': arm_id, 'checkpoint': run_id, 'cwd': str(runtime),
                             'argv': argv, 'output_dir': str(case),
                             'activation': arm['activation'], 'executed': False})
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'commands': commands}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Prepared {len(commands)} commands in {args.output}; no experiments executed.')


if __name__ == '__main__':
    main()
