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
    capabilities = native.capabilities()
    assert capabilities["gpu_interchange"] is False
    assert capabilities["host_vjp"] is False
    assert capabilities["copy_policy"] == "strict contiguous float32; no implicit copy"
    try: native.jvp("add", values([1]), values([2]), values([1]), values([1, 2]))
    except ValueError: pass
    else: raise AssertionError("length mismatch accepted")
    try: native.jvp("add", memoryview(bytearray(4)), right, dl, dr)
    except TypeError: pass
    else: raise AssertionError("bad dtype accepted")
    # The buffers stay retained while native work executes after the GIL drops.
    output = native.jvp("tanh", values([.5]), values([0]), values([.25]), values([0])); gc.collect(); assert len(output) == 1
    strided = memoryview(values([1, 2]))[::2]
    try: native.jvp("add", strided, right, dl, dr)
    except TypeError: pass
    else: raise AssertionError("strided input accepted")
    print("python binding tests passed")
if __name__ == "__main__": main()
