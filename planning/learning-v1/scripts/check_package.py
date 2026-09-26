#!/usr/bin/env python3
"""Read-only package integrity and design lint; not the canonical Todo validator.

Python 3.10+, standard library only. Does not call project-control, modify a
repository or create/update any Todo authority. Use the installed native
validator for actual target validation before adoption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

class PackageError(ValueError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise PackageError(message)

def safe_path(value: str) -> None:
    p=PurePosixPath(value)
    require(bool(value) and not p.is_absolute() and '..' not in p.parts and '\\' not in value,
            f'Unsafe relative path: {value!r}')

def native_digest(plan: dict) -> str:
    # Exact serialization used by the inspected plan_preview implementation.
    return hashlib.sha256(json.dumps(plan, sort_keys=True, indent=2).encode('utf-8')).hexdigest()

def lint_plan(plan: dict, expected_prefix: str) -> dict:
    require(plan.get('schema_version') == 3, 'Native plan must use schema v3')
    tasks=plan.get('tasks',[])
    ids=[t.get('id') for t in tasks]
    require(bool(ids) and all(isinstance(i,str) for i in ids), 'Task IDs required')
    require(len(set(ids))==len(ids), 'Duplicate task IDs')
    require(all(i.startswith(expected_prefix+'-') for i in ids), 'Foreign task identity in native plan')
    by_id={t['id']:t for t in tasks}
    epics=[t for t in tasks if t.get('kind')=='epic']
    require(len(epics)==1, 'Exactly one closure epic is expected')
    epic=epics[0]['id']
    graph={i:[] for i in ids}
    for task in tasks:
        require(task.get('title') and task.get('objective'), f'Missing intent: {task["id"]}')
        require(task.get('completion_contract',{}).get('required'), f'Missing acceptance: {task["id"]}')
        if task['id']!=epic:
            require(task.get('parent_id')==epic,'Outcome must belong to the local epic')
        require('status' not in task and 'result' not in task, 'Bootstrap cannot claim completed work')
        for mode in ('exclusive_paths','read_paths','forbidden_paths'):
            for path in task.get('scope',{}).get(mode,[]): safe_path(path)
        require(not any(p in ('.','.todo-orchestrator','preprint') for p in task.get('scope',{}).get('exclusive_paths',[])), 'Broad/protected ownership not allowed')
        for dep in task.get('depends_on',[]):
            require(dep.get('type')=='task' and dep.get('task_id') in by_id, 'Foreign or unsupported dependency')
            require(dep['task_id']!=epic,'An outcome cannot wait for its own closure epic')
            graph[task['id']].append(dep['task_id'])
        require(not task.get('gates'), 'This package must not invent future product gate commands')
    active=set(); done=set()
    def visit(node: str) -> None:
        require(node not in active,'Dependency cycle')
        if node in done:return
        active.add(node)
        for dep in graph[node]:visit(dep)
        active.remove(node);done.add(node)
    for node in ids:visit(node)
    runs=plan.get('runs',[])
    require(len(runs)==1,'Expected one new local run')
    run=runs[0]
    require(run.get('root_task_id')==epic and isinstance(run.get('charter'),dict),'Invalid run root/charter')
    lanes=run.get('lanes',[])
    require(len(lanes)==1 and lanes[0].get('role')=='implementer','Default must be one usable implementer lane')
    require(lanes[0].get('workspace',{}).get('mode')=='exclusive','Unexpected default workspace')
    queue=lanes[0].get('tasks',[])
    require(len(queue)==len(ids) and set(queue)==set(ids),'Tasks not assigned exactly once')
    require(queue[-1]==epic,'Aggregate must be last')
    position={i:n for n,i in enumerate(queue)}
    for node,deps in graph.items():
        require(all(position[d]<position[node] for d in deps),'Serial queue orders a task before its prerequisite')
    # Approximate generated task brief using the inspected native importer fields.
    sizes=[]
    for task in tasks:
        brief={k:task[k] for k in ('objective','next_action','scope','completion_contract','motivation','desired_end_state','rationale','uncertainties','risks','delegated_choices','references') if k in task}
        sizes.append(len(json.dumps(brief,separators=(',',':'),ensure_ascii=False).encode()))
    require(max(sizes)<6144,'Task context is too large for compact workflow use')
    return {'run':run['id'],'tasks':len(tasks),'outcomes':len(tasks)-1,'lanes':1,
            'maximum_task_brief_bytes':max(sizes),'plan_preview_digest':native_digest(plan)}

def check(root: Path) -> dict:
    root=root.resolve()
    manifest=json.loads((root/'MANIFEST.json').read_text())
    require(manifest.get('format')=='glasshelix-learning-bootstrap','Wrong package manifest')
    entries=manifest.get('files',{})
    for name,expected in entries.items():
        safe_path(name)
        file=root/name
        require(not file.is_symlink(),f'Unexpected symlink: {name}')
        require(file.is_file() and file.resolve().is_relative_to(root),f'Missing or escaping file: {name}')
        require(hashlib.sha256(file.read_bytes()).hexdigest()==expected,f'Integrity mismatch: {name}')
    reports={}
    for filename,prefix in [('glasshelix.todo-plan.json','GH-L1'),('cellerator.optional.todo-plan.json','CE-L1')]:
        plan=json.loads((root/'machine'/filename).read_text())
        reports[filename]=lint_plan(plan,prefix)
        for task in plan['tasks']:
            for ref in task.get('references',[]):
                if ref.startswith('planning/learning-v1/'):
                    local=ref[len('planning/learning-v1/'):]
                    require((root/local).is_file(),f'Missing package reference: {ref}')
    receipt=json.loads((root/'validation/native_preview.json').read_text())
    for key,result in receipt['plans'].items():
        require(reports[key]['plan_preview_digest']==result['plan_digest'],'Plan differs from recorded preview payload')
    return {'ok':True,'scope':'Package hashes, native-plan structural lint and receipt/payload correspondence only. Not implementation or live-authority acceptance.',
            'hashed_files':len(entries),'plans':reports,'native_preview':receipt['plans']}

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try:
        result=check(args.root)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        print(json.dumps({'ok':False,'error':str(exc)},indent=2));return 1
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':
    raise SystemExit(main())
