from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def _atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def experiment_run(base, phase):
    """Publish immutable files and advance the pointer only after a successful body."""
    base = Path(base)
    output = base/'runs'/uuid4().hex
    output.mkdir(parents=True)
    try:
        yield output
        files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.iterdir()) if p.is_file()}
        if not files:
            raise ValueError('Cannot complete an empty experiment')
        manifest = {'run_id':output.name,'phase':phase,'status':'complete','files':files}
        _atomic_json(output/'COMPLETE.json',manifest)
        pointer = 'latest_development.json' if phase=='development' else 'latest_complete.json'
        _atomic_json(base/pointer,{'run_id':output.name,
                                 'manifest_sha256':hashlib.sha256((output/'COMPLETE.json').read_bytes()).hexdigest()})
        print(f'Completed {phase} output: {output}',flush=True)
    except BaseException as error:
        if not (output/'COMPLETE.json').exists():
            _atomic_json(output/'FAILED.json',{'status':'failed','error_type':type(error).__name__})
        raise


def resolve_run(base, phase=None):
    """Verify a completed run; use legacy flat artifacts only when no pointer exists."""
    base = Path(base)
    pointer_path = base/('latest_development.json' if phase=='development' else 'latest_complete.json')
    if not pointer_path.exists():
        return base
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    run_id = pointer['run_id']
    output = base/'runs'/run_id
    if output.resolve().parent != (base/'runs').resolve():
        raise ValueError('Invalid experiment run pointer')
    manifest_bytes = (output/'COMPLETE.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != pointer['manifest_sha256']:
        raise ValueError('Experiment manifest checksum mismatch')
    manifest = json.loads(manifest_bytes)
    if (manifest['status']!='complete' or manifest['run_id']!=run_id or
            (phase=='development') != (manifest['phase']=='development')):
        raise ValueError('Experiment phase does not match completion pointer')
    for name,digest in manifest['files'].items():
        path = output/name
        if path.resolve().parent != output.resolve():
            raise ValueError('Invalid experiment file path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Experiment file checksum mismatch: {name}')
    return output


def selection_run(base):
    base = Path(base)
    return resolve_run(base,phase='development') if (base/'latest_development.json').exists() else resolve_run(base)
