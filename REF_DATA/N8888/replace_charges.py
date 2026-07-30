#!/usr/bin/env python3
"""Replace atom charges in a library .lib file using RESP charges from a NWChem .out file.

Usage: run the script and follow prompts. It will ask for the original .lib file,
the NWChem .out file, then an output filename to save the modified .lib file.

Behavior:
- Extracts RESP charges from a block that starts with a line containing "RESP charges" in the .out file.
- Finds the first occurrence of a table header containing ".unit.atoms" in the .lib file and replaces
  the last numeric field on each atom line with the corresponding RESP charge (in order).

The script attempts to be tolerant of whitespace and formatting differences; it will warn if the
number of RESP charges differs from the number of atom lines replaced.
"""

import re
from pathlib import Path
import sys


def extract_resp_charges(out_text):
    """Return list of float charges parsed from NWChem output text.

    Looks for a line containing 'RESP charges' and then parses subsequent lines like:
      1 C   :   -0.08882
    until an empty line or a non-matching line is encountered after collecting charges.
    """
    charges = []
    lines = out_text.splitlines()
    # find header
    header_idx = None
    for i, ln in enumerate(lines):
        if 'resp charges' in ln.lower():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError('Could not find "RESP charges" block in NWChem output')

    for ln in lines[header_idx+1:]:
        if not ln.strip():
            # stop after an empty line once we've collected stuff
            if charges:
                break
            else:
                continue
        # typical form: index, element, ':', number
        # try split on ':' and parse last part
        if ':' in ln:
            rhs = ln.split(':', 1)[1].strip()
            # rhs may include other text; extract first float
            m = re.search(r'([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)', rhs)
            if m:
                charges.append(float(m.group(1)))
                continue
        # fallback: if line starts with a number token
        m2 = re.search(r'^\s*\d+\s+\w+\s+([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)', ln)
        if m2:
            charges.append(float(m2.group(1)))
            continue
        # if we reach a non-matching line after collecting charges, stop
        if charges:
            break

    if not charges:
        raise ValueError('No RESP charges parsed from NWChem output')
    return charges


def replace_charges_in_lib(lib_text, new_charges):
    """Find the first .unit.atoms table in the lib file and replace the last numeric field on each atom line.

    Returns modified text and a dict with stats.
    """
    lines = lib_text.splitlines()
    # find atoms table header (first occurrence of '.unit.atoms')
    atoms_header_idx = None
    for i, ln in enumerate(lines):
        if '.unit.atoms' in ln:
            atoms_header_idx = i
            break
    if atoms_header_idx is None:
        raise ValueError('Could not find a ".unit.atoms" table in the .lib file')

    out_lines = list(lines)
    charge_idx = 0
    replaced = 0

    # process lines after header until next line that starts with '!entry' (next table) or EOF
    for j in range(atoms_header_idx + 1, len(lines)):
        ln = lines[j]
        if ln.startswith('!entry'):
            break
        # match atom data lines which usually begin with a quoted name:  "C" "c3" ... number
        if re.match(r'^\s*"', ln):
            # capture prefix up to the last number and the last number
            m = re.match(r'^(?P<prefix>.*?\s)(?P<charge>[+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s*$', ln)
            if m:
                if charge_idx < len(new_charges):
                    # Format with 6 decimals (matches existing style like 0.087100)
                    new_charge_str = f"{new_charges[charge_idx]:.6f}"
                    out_lines[j] = m.group('prefix') + new_charge_str
                    charge_idx += 1
                    replaced += 1
                else:
                    # no more new charges; leave remaining as-is
                    break
            else:
                # line starts with quote but didn't match pattern; skip
                continue
        else:
            # not an atom line; skip
            continue

    stats = {
        'atoms_header_idx': atoms_header_idx,
        'replaced': replaced,
        'provided_charges': len(new_charges),
    }
    return '\n'.join(out_lines) + '\n', stats


def prompt_path(prompt_text, expected_ext=None):
    while True:
        p = input(prompt_text).strip('"')
        if not p:
            print('Please enter a path.')
            continue
        path = Path(p)
        if expected_ext and path.suffix.lower() != expected_ext.lower():
            print(f'Warning: file does not have expected extension {expected_ext}. Continue? (y/n)')
            if input().lower().startswith('y'):
                return path
            else:
                continue
        if not path.exists():
            print('File not found. Please enter a valid path.')
            continue
        return path


def main():
    print('Replace charges in .lib file using RESP charges from NWChem .out')
    lib_path = prompt_path('Path to original library file (.lib): ', expected_ext='.lib')
    out_path = prompt_path('Path to NWChem output file (.out): ', expected_ext='.out')

    try:
        out_text = out_path.read_text(encoding='utf-8')
    except Exception:
        out_text = out_path.read_text(encoding='latin-1')

    try:
        charges = extract_resp_charges(out_text)
    except Exception as e:
        print('Error parsing RESP charges:', e)
        sys.exit(1)

    try:
        lib_text = lib_path.read_text(encoding='utf-8')
    except Exception:
        lib_text = lib_path.read_text(encoding='latin-1')

    try:
        new_text, stats = replace_charges_in_lib(lib_text, charges)
    except Exception as e:
        print('Error updating .lib file:', e)
        sys.exit(1)

    print(f"Parsed {stats['provided_charges']} RESP charges; replaced {stats['replaced']} atom lines.")
    out_name = input('Enter output filename to save modified .lib (will overwrite if exists): ').strip('"')
    if not out_name:
        print('No output filename provided. Aborting.')
        sys.exit(1)
    out_file = Path(out_name)
    out_file.write_text(new_text, encoding='utf-8')
    print(f'Saved modified file to: {out_file.resolve()}')


if __name__ == '__main__':
    main()
