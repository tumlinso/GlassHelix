"""CPU witnesses for supplied exact refactorings and synthetic residual learning."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from model import LinearModel, LinearRefactoring, EpochGuard, ResidualActor


def rejected(action, exception=ValueError):
    try:
        action()
    except exception:
        return
    raise AssertionError('invalid operation accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    dtype = torch.float64
    leaf = Path(__file__).resolve().parent
    actor_source = leaf.parent / 'actor/model.py'
    spec = importlib.util.spec_from_file_location('accepted_actor_model', actor_source)
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    a = torch.tensor([[-.4, .2], [-.1, -.7]], dtype=dtype)
    readout = torch.tensor([[.8, -.3], [.1, .5]], dtype=dtype)
    old = LinearModel(a, readout)
    transform = torch.tensor([[1., .4], [-.2, 1.]], dtype=dtype)
    migration = LinearRefactoring.conjugate(old, transform, ('s0', 's1'),
                                            ('anonymous-actor:0', 'residual:0'))
    s = torch.randn(6, 2, dtype=dtype)
    q = migration.initialize_state(s)
    torch.testing.assert_close(q, s @ transform.T)
    torch.testing.assert_close(migration.lift_state(q), s)
    torch.testing.assert_close(migration.model(q), old(s) @ transform.T)
    torch.testing.assert_close(migration.model.observe(q), old.observe(s))
    trajectory_errors = []
    for time in (0., .1, 1., 3.):
        before = old.flow(s, time)
        after = migration.model.flow(q, time)
        torch.testing.assert_close(after, before @ transform.T, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(migration.model.observe(after), old.observe(before),
                                   atol=1e-12, rtol=1e-12)
        trajectory_errors.append((migration.lift_state(after) - before).abs().max().item())
    # Compare complete selected response Jacobians, rather than just one value.
    before_response = torch.autograd.functional.jacobian(
        lambda x: old.observe(old.flow(x, .7)), s)
    after_response = torch.autograd.functional.jacobian(
        lambda x: migration.model.observe(migration.model.flow(x @ transform.T, .7)), s)
    torch.testing.assert_close(before_response, after_response, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(lambda x: migration.model.flow(x, .2),
                                   (q.detach().requires_grad_(),))
    assert torch.autograd.gradcheck(lambda law, obs, x: (
        x @ torch.matrix_exp(.2 * law).T) @ obs.T,
        (migration.model.law.detach().requires_grad_(),
         migration.model.readout.detach().requires_grad_(), q.detach().requires_grad_()))
    rejected(lambda: LinearRefactoring.conjugate(old, torch.zeros_like(transform), ('a','b'), ('c','d')))
    rejected(lambda: LinearRefactoring.conjugate(old, transform, ('a','a'), ('c','d')))

    e = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=dtype)
    p = torch.tensor([[.5, .5, 0.], [0., 0., 1.]], dtype=dtype)
    replicated = LinearModel(torch.diag(torch.tensor([-.3, -.3, -.7], dtype=dtype)),
                             torch.tensor([[1., 2., -.5]], dtype=dtype))
    quotient = LinearRefactoring(replicated, e, p, ('copy0','copy1','other'),
                                 ('shared','other'), quotient=True)
    reduced = torch.randn(5, 2, dtype=dtype)
    full = reduced @ e.T
    torch.testing.assert_close(quotient.initialize_state(full), reduced)
    for time in (.1, 1., 3.):
        torch.testing.assert_close(quotient.lift_state(quotient.model.flow(reduced,time)),
                                   replicated.flow(full,time), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(quotient.model.observe(quotient.model.flow(reduced,time)),
                                   replicated.observe(replicated.flow(full,time)))
    adjoint = torch.tensor([[2., 3., 7.]], dtype=dtype)
    torch.testing.assert_close(quotient.reduce_adjoint(adjoint),
                               torch.tensor([[5., 7.]], dtype=dtype))
    perturbation = torch.randn_like(reduced)
    torch.testing.assert_close(((perturbation @ e.T) * adjoint).sum(),
                               (perturbation * quotient.reduce_adjoint(adjoint)).sum())
    bad = full.clone()
    bad[0,0] += .1
    rejected(lambda: quotient.initialize_state(bad))
    noninvariant = LinearModel(torch.diag(torch.tensor([-.3,-.4,-.7],dtype=dtype)),
                               replicated.readout)
    rejected(lambda: LinearRefactoring(noninvariant,e,p,('a','b','c'),('d','e'),quotient=True))
    rejected(lambda: LinearRefactoring(replicated,e,p*2,('a','b','c'),('d','e'),quotient=True))

    guard = EpochGuard(migration.receipt.old_epoch)
    tape = guard.acquire()
    rejected(guard.publish, RuntimeError)
    guard.release(tape)
    assert guard.publish() == migration.receipt.new_epoch
    rejected(lambda: guard.release(tape), RuntimeError)

    # New explicit actor hidden row h=tanh(U q), readout V h. Base owners stay fixed.
    base = migration.model
    base.requires_grad_(False)
    branch = actor.RegrowthBranch(2, 4, 2).double()
    residual = ResidualActor(base, branch, 'anonymous-residual-actor', epoch=3)
    torch.testing.assert_close(residual(q), old.observe(s), atol=1e-12, rtol=1e-12)
    assert residual.hidden(q).shape == (6, 4)
    assert torch.count_nonzero(branch.inside.weight) > 0
    inputs = torch.randn(32, 2, dtype=dtype) * .6
    with torch.no_grad():
        target = base.observe(inputs) + .3 * torch.tanh(inputs @ torch.tensor(
            [[.7,-.2],[.3,.8]],dtype=dtype))
    # Fresh optimizer explicitly discards unsupported transformed/recycled moments.
    optimizer = torch.optim.Adam(branch.parameters(), lr=.04)
    first = (residual(inputs)-target).square().mean().item()
    (residual(inputs)-target).square().mean().backward()
    outgoing = branch.outside.weight.grad.norm().item()
    assert outgoing > 0 and torch.count_nonzero(branch.inside.weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (residual(inputs)-target).square().mean().backward()
    incoming = branch.inside.weight.grad.norm().item()
    assert incoming > 0
    optimizer.step()
    for _ in range(118):
        optimizer.zero_grad(set_to_none=True)
        loss = (residual(inputs)-target).square().mean()
        loss.backward()
        assert all(torch.isfinite(x.grad).all() for x in branch.parameters())
        optimizer.step()
    last = (residual(inputs)-target).square().mean().item()
    assert last < first * .05
    torch.testing.assert_close(base.law, migration.projection @ a @ migration.embedding)
    torch.testing.assert_close(base.readout, readout @ migration.embedding)
    receipt = {
        'status':'passed', 'device':'cpu', 'dtype':'float64', 'torch_version':torch.__version__,
        'migrations':[asdict(migration.receipt), asdict(quotient.receipt)],
        'parameter_correspondence': {'invertible':'A_new=T A_old T^-1; O_new=O_old T^-1',
                                     'quotient':'A_new=P A_old E; O_new=O_old E'},
        'state_correspondence':{'invertible':'q=T s','quotient':'s=E q; q=P s on invariant manifold'},
        'projection_rebuild':'cold rebuild migrated linear law/readout; no native caches in prototype',
        'trajectory_max_abs_errors':trajectory_errors,
        'checks':['linear_flow_and_readout_correspondence','selected_response_jacobians',
                  'input_law_readout_gradcheck','invariant_quotient_flow', 'replica_adjoint_sum',
                  'off_manifold_and_noninvariant_rejection','invalid_maps', 'live_tape_publication_guard',
                  'initial_residual_zero_effect','live_outgoing_then_incoming_gradient','residual_learning'],
        'residual_actor':{'id':residual.actor_id,'epoch':residual.epoch,'hidden_shape':[32,4],
                          'law':'h=tanh(U q); observed=O_new q+V h',
                          'label':'new model family','behavior':'state trajectory law retained; learned readout changes',
                          'optimizer':'fresh Adam for branch only; base law/readout frozen',
                          'steps':120,'loss_before':first,'loss_after':last,
                          'outgoing_gradient_norm':outgoing,'subsequent_incoming_gradient_norm':incoming},
        'source_sha256':{str(path.relative_to(leaf.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (leaf/'model.py',leaf/'check.py',actor_source)},
        'unsupported':['native CE integration','CUDA','nonlinear discovery','biological identification',
                       'general optimizer-coordinate migration','performance'],
        'meaning':'Caller supplied maps; synthetic engineering evidence only. No approximation claimed.'}
    rendered = json.dumps(receipt,indent=2,sort_keys=True)+'\n'
    if args.receipt:
        args.receipt.write_text(rendered)
    print(rendered,end='')


if __name__ == '__main__':
    main()
