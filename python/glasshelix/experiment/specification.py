"""Small typed scientific manifest; numerical execution retains its owners.

Whole alternatives are supplied, never assembled from marginal coordinates.
Replay restores declarations/results, not framework optimizers or native tapes.
"""
from dataclasses import asdict, dataclass
import json
import math

from ..learning import EvidenceView, EvidencePermission, evidence_manifest


def _ids(values, label):
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'{label} requires unique immutable identities')


def _finite(values):
    return isinstance(values, tuple) and all(type(v) in (int, float) and math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Quantity:
    quantity_id: str
    role: str
    extent: int
    units: str

    def validate(self):
        if (not self.quantity_id or self.role not in
                ('state', 'measurement', 'parameter', 'forcing', 'context', 'inferential_information')
                or type(self.extent) is not int or self.extent <= 0 or not self.units):
            raise ValueError('invalid scientific quantity declaration')


@dataclass(frozen=True)
class JointHypothesis:
    hypothesis_id: str
    mechanism_id: str
    joint_state: tuple[float, ...]
    parameter_ids: tuple[str, ...]
    weight: float

    def validate(self):
        if (not isinstance(self.hypothesis_id, str) or not self.hypothesis_id
                or not isinstance(self.mechanism_id, str) or not self.mechanism_id or not self.joint_state
                or not _finite(self.joint_state) or not isinstance(self.parameter_ids, tuple)
                or any(not isinstance(p, str) or not p for p in self.parameter_ids)
                or len(set(self.parameter_ids)) != len(self.parameter_ids)
                or type(self.weight) not in (float, int)
                or not math.isfinite(self.weight) or self.weight < 0):
            raise ValueError('invalid supplied whole hypothesis')


@dataclass(frozen=True)
class Provenance:
    model_id: str
    model_revision: str
    gh_source_commit: str
    ce_source_commit: str
    numerical_policy: str
    structure_epoch: int
    state_generation: int
    parameter_generation: int
    evidence_ids: tuple[str, ...]

    def validate(self):
        if any(not isinstance(v, str) or not v for v in
               (self.model_id, self.model_revision, self.gh_source_commit,
                self.ce_source_commit, self.numerical_policy)):
            raise ValueError('model, source and numerical provenance required')
        if any(type(v) is not int or v < 0 for v in
               (self.structure_epoch, self.state_generation, self.parameter_generation)):
            raise ValueError('nonnegative epoch and value generations required')
        _ids(self.evidence_ids, 'provenance evidence')


@dataclass(frozen=True)
class Prediction:
    hypothesis_id: str
    observable_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentSpecification:
    experiment_id: str
    model_id: str
    quantities: tuple[Quantity, ...]
    hypotheses: tuple[JointHypothesis, ...]
    source_id: str
    audit_evidence_id: str
    input_manifest_ids: tuple[str, ...]
    source_kind: str = 'synthetic_fixture'

    def validate(self):
        if any(not isinstance(v, str) or not v for v in
               (self.experiment_id, self.model_id, self.source_id, self.audit_evidence_id)):
            raise ValueError('experiment, model, source and audit identities required')
        if self.source_kind not in ('synthetic_fixture', 'audited_celltag'):
            raise ValueError('unknown source kind')
        _ids(self.input_manifest_ids, 'input manifests')
        if not isinstance(self.quantities, tuple) or not isinstance(self.hypotheses, tuple):
            raise ValueError('immutable declarations required')
        for q in self.quantities:
            q.validate()
        _ids(tuple(q.quantity_id for q in self.quantities), 'quantities')
        state_extent = sum(q.extent for q in self.quantities if q.role == 'state')
        parameter_ids = {q.quantity_id for q in self.quantities if q.role == 'parameter'}
        for h in self.hypotheses:
            h.validate()
            if len(h.joint_state) != state_extent or not set(h.parameter_ids) <= parameter_ids:
                raise ValueError('whole state extent or parameter declaration differs')
        _ids(tuple(h.hypothesis_id for h in self.hypotheses), 'whole hypotheses')
        total = sum(h.weight for h in self.hypotheses)
        if not math.isfinite(total) or not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('owner-supplied hypothesis weights must sum to one')

    def authorize(self, evidence: EvidenceView, permission: EvidencePermission,
                  purpose: str, observable_id: str, detection_support=None):
        """Use existing split/modality/cutoff rules and freeze a masked manifest.

        Detection support remains a separate declared boolean mask. Missing
        values are excluded by the original evidence_manifest implementation.
        Population snapshots never gain individual histories from this adapter.
        """
        self.validate()
        observable = next((q for q in self.quantities if q.quantity_id == observable_id), None)
        if (observable is None or observable.role != 'measurement'
                or evidence.values.ndim != 2 or evidence.values.shape[1] != observable.extent
                or evidence.source_id != self.source_id):
            raise ValueError('evidence must match the declared measurement and source')
        # The predecessor validates IDs for uniqueness. This seam additionally
        # rejects empty IDs, which cannot be used for replay/provenance.
        _ids(evidence.evidence_ids, 'observation evidence')
        permission.authorize(evidence, purpose)
        manifest = evidence_manifest(evidence)
        if detection_support is not None:
            if (detection_support.dtype != evidence.observed.dtype
                    or detection_support.shape != evidence.values.shape):
                raise ValueError('detection support requires a matching boolean mask')
            manifest['detection_support'] = detection_support.detach().cpu().tolist()
        else:
            manifest['detection_support'] = None
        return {'observable_id': observable_id, 'purpose': purpose,
                'sampling_unit': 'population_snapshot', 'observation': manifest}


@dataclass(frozen=True)
class ScientificResult:
    specification: ExperimentSpecification
    hypotheses: tuple[JointHypothesis, ...]
    predictions: tuple[Prediction, ...]
    provenance: Provenance
    conditioning_evidence_ids: tuple[str, ...] = ()
    interpretation_scope: str = 'supplied_alternatives'
    biological_status: str = 'not_run'

    def validate(self):
        self.specification.validate()
        self.provenance.validate()
        if self.provenance.model_id != self.specification.model_id:
            raise ValueError('result model differs from specification')
        if self.interpretation_scope != 'supplied_alternatives':
            raise ValueError('supplied candidate results do not establish global identification or individual history')
        # This frontier contains engineering fixtures; an audit is not a fit.
        if self.biological_status != 'not_run':
            raise ValueError('biological fitting is unqualified in this scientific seam')
        expected = {h.hypothesis_id: h for h in self.specification.hypotheses}
        if not isinstance(self.hypotheses, tuple) or not isinstance(self.predictions, tuple):
            raise ValueError('immutable whole hypotheses and predictions required')
        _ids(tuple(h.hypothesis_id for h in self.hypotheses), 'result hypotheses')
        if set(expected) != {h.hypothesis_id for h in self.hypotheses}:
            raise ValueError('result must retain every whole hypothesis')
        for h in self.hypotheses:
            h.validate()
            original = expected[h.hypothesis_id]
            if (h.mechanism_id != original.mechanism_id or h.parameter_ids != original.parameter_ids
                    or h.joint_state != original.joint_state):
                raise ValueError('result changed a whole hypothesis declaration')
        if not math.isclose(sum(h.weight for h in self.hypotheses), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('result owner weights must sum to one')
        measurements = {q.quantity_id: q for q in self.specification.quantities if q.role == 'measurement'}
        keys = set()
        for p in self.predictions:
            key = (p.hypothesis_id, p.observable_id)
            observable = measurements.get(p.observable_id)
            if (key in keys or p.hypothesis_id not in expected or observable is None
                    or not _finite(p.values) or len(p.values) != observable.extent):
                raise ValueError('invalid whole-hypothesis measurement prediction')
            keys.add(key)
        if {p.hypothesis_id for p in self.predictions} != set(expected):
            raise ValueError('prediction required for every whole hypothesis')
        if (not isinstance(self.conditioning_evidence_ids, tuple)
                or len(set(self.conditioning_evidence_ids)) != len(self.conditioning_evidence_ids)
                or any(not x or x not in self.provenance.evidence_ids for x in self.conditioning_evidence_ids)):
            raise ValueError('conditioning evidence must be unique and retained in provenance')

    def dumps(self):
        self.validate()
        return json.dumps({'schema': 1, 'result': asdict(self)}, sort_keys=True, allow_nan=False)

    @classmethod
    def loads(cls, text):
        def reject_constant(value):
            raise ValueError(f'nonfinite replay constant: {value}')
        payload = json.loads(text, parse_constant=reject_constant)
        if payload.get('schema') != 1 or set(payload) != {'schema', 'result'}:
            raise ValueError('unsupported scientific replay schema')
        raw = payload['result']
        spec = dict(raw['specification'])
        spec['quantities'] = tuple(Quantity(**q) for q in spec['quantities'])
        def hypothesis(h):
            return JointHypothesis(**dict(h, joint_state=tuple(h['joint_state']),
                                          parameter_ids=tuple(h['parameter_ids'])))
        spec['hypotheses'] = tuple(hypothesis(h) for h in spec['hypotheses'])
        spec['input_manifest_ids'] = tuple(spec['input_manifest_ids'])
        provenance = dict(raw['provenance'])
        provenance['evidence_ids'] = tuple(provenance['evidence_ids'])
        result = cls(ExperimentSpecification(**spec), tuple(hypothesis(h) for h in raw['hypotheses']),
                     tuple(Prediction(**dict(p, values=tuple(p['values']))) for p in raw['predictions']),
                     Provenance(**provenance), tuple(raw['conditioning_evidence_ids']),
                     raw['interpretation_scope'], raw['biological_status'])
        result.validate()
        return result
