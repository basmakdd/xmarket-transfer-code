"""Parse the auto-generated LaTeX tables so figures never use hand-typed numbers."""
import re, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'paper'))       # legacy_v3/paper
RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'paper', 'raw'))  # shipped results

def _read(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        return f.read()

def _num(s):
    s = s.strip().replace(',', '').replace('$', '')
    if s in ('--', ''):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return s

def block(texname, label):
    """Return data rows (list of cell lists) of the table carrying \\label{label}."""
    tex = _read(texname)
    i = tex.index('\\label{%s}' % label)
    # table ends at the next \bottomrule
    j = tex.index('\\bottomrule', i)
    body = tex[i:j]
    # data rows: after last \endlastfoot or after first \midrule
    if '\\endlastfoot' in body:
        body = body.split('\\endlastfoot', 1)[1]
    else:
        body = body.split('\\midrule', 1)[1]
    rows = []
    for line in body.split('\n'):
        line = line.strip()
        if not line or line.startswith('\\midrule') or line.startswith('\\multicolumn'):
            continue
        line = line.replace('\\&', '&&AMP&&').rstrip('\\').strip()
        cells = [c.replace('&&AMP&&', '&').strip() for c in re.split(r'(?<!&)&(?!&)', line.replace('&&AMP&&', '\x00'))]
        cells = [c.replace('\x00', '&') for c in cells]
        cells = [re.sub(r'^\\multicolumn\{\d+\}\{\w\}\{(.*)\}$', r'\1', c) for c in cells]
        rows.append([_num(c) for c in cells])
    return rows

def ci(s):
    """'+0.0251 [+0.003 +0.071]' -> (0.0251, 0.003, 0.071)"""
    v = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', str(s))]
    return tuple(v[:3])

def detail(label):
    return block('tables_detailed.tex', label)

def main(label):
    return block('tables.tex', label)
