"""Run a project script with stable imports and project-relative data paths."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'

def main():
    args = sys.argv[1:]
    if not args or args[0] in ('--list', '-h', '--help'):
        print('Usage: python run.py <script_name> [arguments]')
        print('\nAvailable scripts:')
        print('\n'.join(p.stem for p in sorted(SCRIPTS.glob('*.py'))))
        return
    name = args[0]
    if name.endswith('.py'):
        name = name[:-3]
    path = SCRIPTS / (name + '.py')
    if Path(name).name != name or not path.is_file() or path.resolve().parent != SCRIPTS.resolve():
        raise SystemExit('Unknown script. Use python run.py --list.')
    os.chdir(ROOT)
    sys.path.insert(0, str(SCRIPTS))
    sys.argv = [str(path), *args[1:]]
    runpy.run_path(str(path), run_name='__main__')

if __name__ == '__main__':
    main()
