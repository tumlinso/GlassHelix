"""One bounded framework experiment and scientific result evaluation/replay."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def _read(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _write(path, value):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, allow_nan=False)
    Path(path).write_text(text + '\n')


def capabilities():
    packages = {name: importlib.util.find_spec('glasshelix.' + name) is not None
                for name in ('experiment', 'data', 'models', 'analysis', 'refactoring')}
    framework = importlib.util.find_spec('torch') is not None
    native = importlib.util.find_spec('glasshelix_nf1') is not None
    return {'components': packages, 'framework_dependency': 'available' if framework else 'missing torch',
            'run': 'available' if framework and packages['experiment'] and packages['models'] else 'unavailable',
            'evaluate_replay': 'available' if framework and packages['experiment'] else 'unavailable',
            'native_local_arithmetic': 'provider_present' if native else 'unavailable',
            'native_epoch_publication': 'unqualified', 'biology': 'not_run'}


def run_plan(plan):
    import torch
    from glasshelix.experiment import (ExperimentSpecification, JointHypothesis, Quantity,
                                      Prediction, Provenance, ScientificResult)
    from glasshelix.learning import ProductHypothesis
    from glasshelix.models import ModelClient, ModelDeclaration
    if plan.get('driver') != 'joint_product_reference':
        raise NotImplementedError('this entry point supports the declared joint product reference driver')
    coefficient = plan['coefficient']
    if type(coefficient) not in (int, float) or not math.isfinite(coefficient):
        raise ValueError('finite supplied coefficient required')
    raw = dict(plan['specification'])
    raw['quantities'] = tuple(Quantity(**q) for q in raw['quantities'])
    raw['hypotheses'] = tuple(JointHypothesis(**dict(h, joint_state=tuple(h['joint_state']),
                                                   parameter_ids=tuple(h['parameter_ids'])))
                              for h in raw['hypotheses'])
    raw['input_manifest_ids'] = tuple(raw['input_manifest_ids'])
    specification = ExperimentSpecification(**raw)
    specification.validate()
    provenance_raw = dict(plan['provenance'])
    provenance_raw['evidence_ids'] = tuple(provenance_raw['evidence_ids'])
    provenance = Provenance(**provenance_raw)
    if provenance.numerical_policy != 'torch_cpu_f64_reference':
        raise ValueError('executed reference requires its explicit torch_cpu_f64_reference policy')
    observable = next((q for q in specification.quantities if q.quantity_id == plan['observable_id']), None)
    if observable is None or observable.role != 'measurement' or observable.extent != 1:
        raise ValueError('joint product driver requires one declared measured response')
    owner = ProductHypothesis(initial=coefficient, dtype=torch.float64)
    predictions = []
    for h in specification.hypotheses:
        if len(h.joint_state) != 2 or h.parameter_ids != (plan['coefficient_id'],):
            raise ValueError('joint product driver requires two joint state values and one shared coefficient ID')
        declaration = ModelDeclaration(specification.model_id, h.hypothesis_id, h.mechanism_id,
            'joint_product', (2,), ('declared-joint-state',),
            (('coefficients', plan['coefficient_id']),), plan['provider_evidence_id'], True)
        client = ModelClient(owner, declaration)
        state = torch.tensor([h.joint_state], dtype=torch.float64)
        predictions.append(client.to_prediction(client(state), plan['observable_id']))
    result = ScientificResult(specification, specification.hypotheses, tuple(predictions), provenance)
    result.validate()
    return result.dumps()


def evaluate_result(result_text, observation):
    import torch
    from glasshelix.experiment import ScientificResult
    from glasshelix.learning import EvidenceView, masked_loss
    result = ScientificResult.loads(result_text)
    values, present = observation['values'], observation['present']
    if any(type(x) is not bool for x in present):
        raise ValueError('observation missingness requires boolean presence values')
    evidence = EvidenceView(torch.tensor([values], dtype=torch.float64),
        torch.tensor([present], dtype=torch.bool), (observation['evidence_id'],),
        observation['modality'], observation['split'], (observation['time'],), observation['source_id'])
    evidence.validate()
    if evidence.split not in ('validation', 'test'):
        raise ValueError('evaluation requires a declared held-out split')
    scores = []
    for prediction in result.predictions:
        if prediction.observable_id == observation['observable_id']:
            scores.append({'hypothesis_id': prediction.hypothesis_id,
                           'masked_loss': float(masked_loss(torch.tensor([prediction.values],
                                                                        dtype=torch.float64), evidence))})
    if not scores:
        raise ValueError('result lacks the declared measurement prediction')
    return {'experiment_id': result.specification.experiment_id, 'evidence_id': observation['evidence_id'],
            'sampling_unit': 'population_snapshot', 'scope': 'local supplied candidate scores',
            'scores': scores, 'biological_status': result.biological_status}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glasshelix-substrate')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('capabilities')
    commands.add_parser('native-capabilities')
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--output', required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--result', required=True)
    evaluate.add_argument('--observations', required=True)
    evaluate.add_argument('--output', required=True)
    replay = commands.add_parser('replay')
    replay.add_argument('--result', required=True)
    replay.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'capabilities':
            print(json.dumps(capabilities(), sort_keys=True))
        elif args.command == 'native-capabilities':
            from glasshelix.nf1 import capabilities as native_capabilities
            print(json.dumps(native_capabilities(), sort_keys=True))
        elif args.command == 'run':
            _write(args.output, run_plan(_read(args.plan)))
        elif args.command == 'evaluate':
            _write(args.output, evaluate_result(Path(args.result).read_text(), _read(args.observations)))
        else:
            from glasshelix.experiment import ScientificResult
            _write(args.output, ScientificResult.loads(Path(args.result).read_text()).dumps())
    except (ImportError, ValueError, KeyError, TypeError, NotImplementedError, OSError) as error:
        parser.exit(2, f'{args.command} unavailable or invalid: {error}\n')
    return 0
