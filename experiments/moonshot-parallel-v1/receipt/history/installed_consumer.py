#!/usr/bin/env python3
"""CPU FP32 installed product2 consumer; independent scalar reference."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = pathlib.Path(manifest['source_root'])
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == manifest['source_commit']
    installed = pathlib.Path(manifest['install_target'])
    for relative, digest in manifest['package_files_sha256'].items():
        assert sha(installed / relative) == digest
        assert sha(source / 'components/CelleraTorch/python' / relative) == digest
    assert sha(manifest['native_library']) == manifest['native_library_sha256']
    import os
    os.environ['CELLERATOR_PRODUCT2_LIBRARY'] = manifest['native_library']
    sys.path.insert(0, str(installed))
    import torch
    import celleratorch.moonshot_product as adapter
    assert pathlib.Path(adapter.__file__).resolve() == (installed / 'celleratorch/moonshot_product.py').resolve()
    x = torch.tensor([0., 2., -1.], requires_grad=True)
    k = torch.tensor([1.5, -2., .5, 3.], requires_grad=True)
    a = torch.tensor([0, 1, 1, 2]); b = torch.tensor([1, 1, 2, 0])
    g = torch.tensor([.4, -.7, .2, 1.1])
    dx = torch.tensor([.3, -.2, .7]); dk = torch.tensor([.1, .2, -.3, .4])
    expected_y=[]; expected_gx=[0.]*3; expected_gk=[]; expected_jvp=[]
    for i in range(4):
        u,v=int(a[i]),int(b[i]); xi,xj,ki=float(x[u].detach()),float(x[v].detach()),float(k[i].detach())
        expected_y.append(ki*xi*xj)
        expected_gx[u]+=float(g[i])*ki*xj
        expected_gx[v]+=float(g[i])*ki*xi
        expected_gk.append(float(g[i])*xi*xj)
        expected_jvp.append(float(dk[i])*xi*xj+ki*(float(dx[u])*xj+xi*float(dx[v])))
    y=adapter.product2(x,k,a,b); gx,gk=torch.autograd.grad(y,(x,k),g)
    primal,jvp=adapter.product2_jvp(x,k,a,b,dx,dk)
    for actual,expected in ((y,expected_y),(gx,expected_gx),(gk,expected_gk),(jvp,expected_jvp)):
        torch.testing.assert_close(actual,torch.tensor(expected),rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(primal,y.detach())
    adjoint_error=abs(float((jvp*g).sum()-(gx*dx).sum()-(gk*dk).sum()))
    assert adjoint_error < 2e-6
    eps=1e-3
    def scalar_reference(xx,kk):
        return torch.tensor([float(kk[i])*float(xx[int(a[i])])*float(xx[int(b[i])]) for i in range(4)])
    finite_difference=(scalar_reference(x.detach()+eps*dx,k.detach()+eps*dk)-scalar_reference(x.detach()-eps*dx,k.detach()-eps*dk))/(2*eps)
    torch.testing.assert_close(jvp,finite_difference,rtol=5e-4,atol=5e-4)
    module=adapter.Product2Module(k.detach(),a,b)
    torch.testing.assert_close(module(x.detach()),y.detach())
    module_jvp=module.jvp(x.detach(),dx,dk)[1]
    torch.testing.assert_close(module_jvp,jvp)
    # One external optimizer step must consume the native parameter VJP.
    opt=torch.optim.SGD(module.parameters(),lr=.01)
    before=module.coefficients.detach().clone()
    (module(x.detach())*g).sum().backward(); opt.step()
    torch.testing.assert_close(module.coefficients,before-.01*gk)
    result={'schema_version':1,'status':'passed','source_commit':manifest['source_commit'],
            'consumer_sha256':sha(__file__),'install_manifest_sha256':sha(args.manifest),
            'adapter_file':adapter.__file__,'adapter_sha256':sha(adapter.__file__),
            'native_library':manifest['native_library'],'native_library_sha256':sha(manifest['native_library']),
            'python':sys.executable,'torch_version':torch.__version__,'device':'cpu','precision':'float32',
            'fixture':{'x':x.detach().tolist(),'k':k.detach().tolist(),'a':a.tolist(),'b':b.tolist(),'cotangent':g.tolist(),'dx':dx.tolist(),'dk':dk.tolist()},
            'results':{'forward':y.detach().tolist(),'input_vjp':gx.tolist(),'parameter_vjp':gk.tolist(),'jvp':jvp.tolist(),'adjoint_error':adjoint_error},
            'checks':['independent scalar forward','repeated-index and zero input VJP','parameter VJP','full input and parameter JVP','finite difference','adjoint identity','installed Product2Module','external SGD consumes native VJP'],
            'unsupported':['CUDA Torch tensors','mixed precision','second-order transforms','biological inference']}
    for relative,digest in manifest['package_files_sha256'].items(): assert sha(installed/relative)==digest
    assert sha(manifest['native_library'])==manifest['native_library_sha256']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Installed native CPU product2 consumer: 8 checks passed')

if __name__ == '__main__':
    main()
