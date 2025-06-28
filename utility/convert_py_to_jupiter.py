#!/usr/bin/env python3
"""
convert_py_to_ipynb.py

Scans a target directory (default: "notebooks") for .py files. For each .py file:
  - Parses it into cells by looking for lines starting with "# %%".
    - If the marker is "# %% [markdown]", the following lines until the next marker
      are treated as a markdown cell (stripping leading "# ").
    - Otherwise (just "# %%" or "# %% [something else]"), it is a code cell.
  - Builds a new .ipynb with the same basename.
  - Deletes the original .py file after successful conversion.
Usage:
    python convert_py_to_ipynb.py [--dir PATH_TO_DIR]
Dependencies:
    pip install nbformat
"""

import argparse
import os
import re
import nbformat
from pathlib import Path
import sys

def parse_py_to_cells(py_path):
    """
    Reads a .py file and splits it into cells based on "# %%" markers.
    Returns a list of tuples: (cell_type, source_str).
    cell_type is "code" or "markdown".
    For markdown cells, leading "# " or "#" is stripped from each line.
    If no markers are found, the entire file is one code cell.
    """
    marker_re = re.compile(r'^\s*#\s*%%')  # matches "# %%", possibly with leading whitespace
    markdown_marker_re = re.compile(r'^\s*#\s*%%\s*\[markdown\]', re.IGNORECASE)

    lines = py_path.read_text(encoding='utf-8').splitlines(keepends=True)
    cells = []
    current_lines = []
    current_type = 'code'  # default until first marker

    saw_any_marker = False

    for line in lines:
        if marker_re.match(line):
            saw_any_marker = True
            # On encountering a marker: finish previous cell (if any)
            if current_lines:
                src = ''.join(current_lines).rstrip('\n')
                cells.append((current_type, src))
            # Determine new cell type
            if markdown_marker_re.match(line):
                current_type = 'markdown'
            else:
                current_type = 'code'
            current_lines = []
            # Skip the marker line itself (do not include in cell content)
        else:
            # Append line to current cell, with appropriate stripping if markdown
            if current_type == 'markdown':
                # Strip leading "#" and optional one space
                # e.g. "# Hello" -> "Hello", "   # Hello" -> "Hello"
                stripped = re.sub(r'^\s*#\s?', '', line)
                current_lines.append(stripped)
            else:
                current_lines.append(line)
    # After loop, append any remaining lines
    if current_lines:
        src = ''.join(current_lines).rstrip('\n')
        cells.append((current_type, src))

    # If no markers were found, treat entire file as one code cell
    if not saw_any_marker:
        entire = py_path.read_text(encoding='utf-8')
        return [('code', entire.rstrip('\n'))]
    return cells

def convert_file(py_path, overwrite_ipynb=True):
    """
    Converts a single .py file at py_path into a .ipynb in the same directory.
    If overwrite_ipynb is False and .ipynb exists, skip conversion.
    Returns True if conversion succeeded (and we can delete .py), False otherwise.
    """
    nb_path = py_path.with_suffix('.ipynb')
    if nb_path.exists() and not overwrite_ipynb:
        print(f"Skipping {py_path.name}: {nb_path.name} already exists")
        return False

    try:
        cells = parse_py_to_cells(py_path)
        nb = nbformat.v4.new_notebook()
        nb.metadata.setdefault('kernelspec', {
            "name": "python3",
            "display_name": "Python 3"
        })
        nb.metadata.setdefault('language_info', {
            "name": "python",
            "file_extension": ".py"
        })

        nb_cells = []
        for cell_type, src in cells:
            if not src.strip():
                continue  # skip empty cells
            if cell_type == 'markdown':
                nb_cells.append(nbformat.v4.new_markdown_cell(source=src))
            else:
                nb_cells.append(nbformat.v4.new_code_cell(source=src))
        nb['cells'] = nb_cells

        # Write notebook
        with nb_path.open('w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Converted: {py_path.name} -> {nb_path.name}")
        return True
    except Exception as e:
        print(f"Error converting {py_path}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert .py with # %% cells into Jupyter notebooks.")
    parser.add_argument('--dir', type=str, default='../notebooks',
                        help='Directory to scan for .py files (default: "../../notebooks")')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='If set, do not overwrite existing .ipynb files; skip them instead.')
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.is_dir():
        print(f"Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    py_files = list(target_dir.glob('*.py'))
    if not py_files:
        print(f"No .py files found in {target_dir}")
        return

    for py_path in py_files:
        succeeded = convert_file(py_path, overwrite_ipynb=not args.no_overwrite)
        if succeeded:
            try:
                py_path.unlink()
                print(f"Deleted original: {py_path.name}")
            except Exception as e:
                print(f"Warning: could not delete {py_path.name}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
