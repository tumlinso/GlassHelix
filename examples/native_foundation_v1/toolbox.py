"""Prospective Python parity consumer. Requires the completed real native bindings."""
import numpy as np
import glasshelix as gh

def main():
    # This composition uses the same native builder and executor as the C++ program.
    s = gh.Session(backend="cuda")
    b = gh.SystemBuilder()
    x = [b.state(name) for name in ("activity-0", "activity-1", "hidden-activity", "observed-output")]
    a, beta = b.parameter_value("a"), b.parameter_value("b")
    p, q, r = (b.argument(i) for i in range(3))
    triad = b.nary("joint-triad", x[:3], gh.tanh(p) * (q / (1 + gh.square(q))) * r)
    b.derivative(x[0], b.forcing("external-input") - .7 * b.read(x[0]))
    b.derivative(x[1], .4 * b.read(x[0]) - .5 * b.read(x[1]))
    b.derivative(x[2], .2 + .3 * b.read(x[1]) - .4 * b.read(x[2]))
    b.derivative(x[3], (b.read(a) + b.read(beta)) * b.result(triad) - .6 * b.read(x[3]))
    observed = b.observe("partial-readout", b.read(x[3]))
    program = gh.PreparedSystem(s, b.finish())
    state = program.instantiate(np.array([.8, .7], dtype=np.float64))
    i = np.arange(33)
    initial = np.stack((.1+.001*i, .2+.002*i, .8+.003*i, np.full(33,.05)), axis=1)
    state.set_initial(initial, batch=33)  # explicit host-to-native conversion/upload boundary
    result = program.rk4(state, 0., 2., .01, np.array([[0.,1.],[.75,0.]], dtype=np.float64))
    result.ready.wait()
    values = result.download(observed)  # explicit completion/readback, not a hidden array conversion
    assert values.shape == (33,)
    assert np.isfinite(values).all()
    print(s.explain())
    return values

if __name__ == "__main__":
    main()
