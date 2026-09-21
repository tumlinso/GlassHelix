import array
import gc
import subprocess
import sys

import glasshelix_nf1 as native

def values(xs): return array.array("f", xs)
def main():
    left, right, dl, dr = values([2, 3]), values([5, 7]), values([.5, .25]), values([.1, .2])
    assert native.jvp("multiply", left, right, dl, dr) == [2.700000047683716, 2.3499999046325684]
    assert native.jvp("add", left, right, dl, dr) == [0.6000000238418579, 0.44999998807907104]
    assert native.capabilities()["gpu_interchange"] is False
    try: native.jvp("add", values([1]), values([2]), values([1]), values([1, 2]))
    except ValueError: pass
    else: raise AssertionError("length mismatch accepted")
    try: native.jvp("add", memoryview(bytearray(4)), right, dl, dr)
    except TypeError: pass
    else: raise AssertionError("bad dtype accepted")
    # The buffers stay retained while the native callback executes; this probes
    # a temporary input after its Python name has been dropped.
    output = native.jvp("tanh", values([.5]), values([0]), values([.25]), values([0])); gc.collect(); assert len(output) == 1
    print("python binding tests passed")
if __name__ == "__main__": main()
