"""Execute the notebook with the project interpreter and preserve outputs.

Use --development to validate the independent exploration/fit cells first.
"""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    path = Path('value-at-risk-modelling.ipynb')
    notebook = nbformat.read(path, as_version=4)
    if args.development:
        notebook.cells = notebook.cells[:15] + notebook.cells[19:21] + notebook.cells[25:27]
        path = Path('results/development_validation.ipynb')
    # Supply an explicit interpreter rather than relying on a global kernelspec.
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [str(Path('.venv/bin/python').absolute()), '-m', 'ipykernel_launcher',
                                '-f', '{connection_file}']
    manager.kernel_spec.env = {'MPLCONFIGDIR': '/tmp/var-matplotlib',
                               'XDG_CACHE_HOME': '/tmp/var-cache',
                               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:90]}', flush=True)

    client = NotebookClient(notebook, km=manager, timeout=7200,
                            resources={'metadata': {'path': str(Path.cwd())}},
                            on_cell_start=progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print(f'Executed and saved {path}', flush=True)


if __name__ == '__main__':
    main()
