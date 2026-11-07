"""
filter_effect_table.py
----------------------
Generates the filter-effect table from task_performance_summary.csv:
one row per (rat, condition) cell whose tier changed between the
unfiltered and filtered classifications.

Outputs:
    <OUTPUT_DIR>/tables/filter_effect_table.tex   (booktabs LaTeX)
    <OUTPUT_DIR>/tables/filter_effect_table.csv   (same content, plain CSV)

Also prints a plain-text version to the console for eyeballing.
"""

import argparse
from pathlib import Path
import pandas as pd


SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / 'results' / 'task_check_output_final' / 'task_performance_summary.csv'
DEFAULT_OUT = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_1")

TIER_RANK = {'clean': 0, 'usable': 1, 'poor': 2}


def _pivot(df, col):
    return df.pivot_table(index=['rat', 'condition'], columns='filter_state',
                          values=col, aggfunc='first')


def build_table(csv_path):
    df = pd.read_csv(csv_path)
    p_cls = _pivot(df, 'classification')
    p_acc = _pivot(df, 'accuracy')
    p_sig = _pivot(df, 'sigma_97_5')
    p_lap = _pivot(df, 'lapse_sum')
    p_n   = _pivot(df, 'n_trials')

    changed = p_cls[p_cls['unfiltered'] != p_cls['filtered']].reset_index()
    rows = []
    for _, r in changed.iterrows():
        key = (r['rat'], r['condition'])
        ru = TIER_RANK[p_cls.loc[key, 'unfiltered']]
        rf = TIER_RANK[p_cls.loc[key, 'filtered']]
        direction = 'up' if rf < ru else 'down'
        dn = 100 * (p_n.loc[key, 'filtered'] - p_n.loc[key, 'unfiltered']) / p_n.loc[key, 'unfiltered']
        rows.append({
            'direction':  direction,
            'rat':        int(r['rat']),
            'condition':  str(r['condition']),
            'trans':      f"{p_cls.loc[key,'unfiltered']} → {p_cls.loc[key,'filtered']}",
            'n_u':        int(p_n.loc[key, 'unfiltered']),
            'n_f':        int(p_n.loc[key, 'filtered']),
            'dn_pct':     dn,
            'acc_u':      100 * p_acc.loc[key, 'unfiltered'],
            'acc_f':      100 * p_acc.loc[key, 'filtered'],
            'sig_u':      p_sig.loc[key, 'unfiltered'],
            'sig_f':      p_sig.loc[key, 'filtered'],
            'lap_u':      p_lap.loc[key, 'unfiltered'],
            'lap_f':      p_lap.loc[key, 'filtered'],
        })

    # order: upgrades first (by rat, cond), then downgrades
    tbl = pd.DataFrame(rows)
    tbl['_dir_order'] = tbl['direction'].map({'up': 0, 'down': 1})
    tbl = tbl.sort_values(['_dir_order', 'rat', 'condition']).drop(columns='_dir_order')
    return tbl


def format_condition(cond):
    """Turn '1', '2', '3', 'pooled' into pretty labels."""
    if cond == 'pooled':
        return 'Pooled'
    return f'Mod {cond}'


def write_latex(tbl, path):
    """Booktabs LaTeX with upgrades and downgrades split by a midrule."""
    n_up   = int((tbl['direction'] == 'up').sum())
    n_down = int((tbl['direction'] == 'down').sum())

    lines = []
    lines.append(r"% Requires \usepackage{booktabs} (and textcomp for \textdegree).")
    lines.append(r"\begin{table}[tbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \begin{tabular}{llllrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    & Rat & Cond. & Tier transition & $\Delta n$ (\%) "
                 r"& Acc.\ $u \rightarrow f$ (\%) "
                 r"& $\sigma_{97.5}\ u \rightarrow f$ (\textdegree) "
                 r"& $\gamma+\lambda\ u \rightarrow f$ \\")
    lines.append(r"    \midrule")

    # upgrades
    lines.append(rf"    \multicolumn{{8}}{{l}}{{\emph{{Upgrades ({n_up})}}}} \\")
    for _, r in tbl[tbl['direction'] == 'up'].iterrows():
        arrow = r'$\uparrow$'
        lines.append(
            f"    {arrow} & {r['rat']} & {format_condition(r['condition'])} & "
            f"{r['trans']} & {r['dn_pct']:+.0f} & "
            f"{r['acc_u']:.1f} $\\rightarrow$ {r['acc_f']:.1f} & "
            f"{r['sig_u']:.1f} $\\rightarrow$ {r['sig_f']:.1f} & "
            f"{r['lap_u']:.3f} $\\rightarrow$ {r['lap_f']:.3f} \\\\"
        )

    lines.append(r"    \midrule")

    # downgrades
    lines.append(rf"    \multicolumn{{8}}{{l}}{{\emph{{Downgrades ({n_down})}}}} \\")
    # (column count matches the 8-column tabular above)
    for _, r in tbl[tbl['direction'] == 'down'].iterrows():
        arrow = r'$\downarrow$'
        lines.append(
            f"    {arrow} & {r['rat']} & {format_condition(r['condition'])} & "
            f"{r['trans']} & {r['dn_pct']:+.0f} & "
            f"{r['acc_u']:.1f} $\\rightarrow$ {r['acc_f']:.1f} & "
            f"{r['sig_u']:.1f} $\\rightarrow$ {r['sig_f']:.1f} & "
            f"{r['lap_u']:.3f} $\\rightarrow$ {r['lap_f']:.3f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(
        r"  \caption{Cells whose overall performance tier changed after "
        r"applying the training-phase filter. Only the "
        rf"{n_up + n_down} of 48 (rat $\times$ condition) cells whose overall "
        r"tier changed are shown; the remaining 40 cells kept the same "
        r"classification in both filter states. For each cell we report the "
        r"tier transition, the percentage change in trial count "
        r"($\Delta n$), and the values of the three fit-derived summary "
        r"parameters --- accuracy on unambiguous trials, the 97.5-percentile "
        r"of the session-bootstrap distribution of $\sigma$, and the total "
        r"lapse rate $\gamma+\lambda$ --- in the unfiltered ($u$) and "
        r"filtered ($f$) states. Every downgrade is driven by "
        r"$\sigma_{97.5}$ crossing the $35\degree$ upper bound of the "
        r"\emph{usable} tier while accuracy remains essentially unchanged."
        r"}"
    )
    lines.append(r"  \label{tab:filter_effect}")
    lines.append(r"\end{table}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_csv(tbl, path):
    out = tbl.copy()
    out['condition'] = out['condition'].map(format_condition)
    out['accuracy_u_pct']       = out['acc_u'].round(1)
    out['accuracy_f_pct']       = out['acc_f'].round(1)
    out['sigma_97_5_u_deg']     = out['sig_u'].round(1)
    out['sigma_97_5_f_deg']     = out['sig_f'].round(1)
    out['gamma_plus_lambda_u']  = out['lap_u'].round(3)
    out['gamma_plus_lambda_f']  = out['lap_f'].round(3)
    out['delta_n_pct']          = out['dn_pct'].round(0).astype(int)
    keep = ['direction', 'rat', 'condition', 'trans', 'n_u', 'n_f', 'delta_n_pct',
            'accuracy_u_pct', 'accuracy_f_pct',
            'sigma_97_5_u_deg', 'sigma_97_5_f_deg',
            'gamma_plus_lambda_u', 'gamma_plus_lambda_f']
    out[keep].to_csv(path, index=False)


def print_console(tbl):
    print("\n" + "=" * 92)
    print(f"FILTER-EFFECT TABLE  ({len(tbl)} cells changed tier)")
    print("=" * 92)
    disp = tbl.copy()
    disp['condition'] = disp['condition'].map(format_condition)
    disp['acc']  = disp.apply(lambda r: f"{r['acc_u']:.1f} → {r['acc_f']:.1f}", axis=1)
    disp['sig']  = disp.apply(lambda r: f"{r['sig_u']:.1f} → {r['sig_f']:.1f}", axis=1)
    disp['lap']  = disp.apply(lambda r: f"{r['lap_u']:.3f} → {r['lap_f']:.3f}", axis=1)
    disp['dn']   = disp['dn_pct'].map(lambda v: f"{v:+.0f}")
    show = disp[['direction', 'rat', 'condition', 'trans', 'dn', 'acc', 'sig', 'lap']]
    show.columns = ['dir', 'rat', 'cond', 'transition', 'Δn(%)',
                    'acc(u→f)', 'σ97.5(u→f)', 'γ+λ(u→f)']
    print(show.to_string(index=False))
    print("=" * 92)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tbl = build_table(args.csv)
    print_console(tbl)

    args.out.mkdir(parents=True, exist_ok=True)
    write_latex(tbl, args.out / 'filter_effect_table.tex')
    write_csv(tbl,   args.out / 'filter_effect_table.csv')
    print(f"\n✓ LaTeX: {args.out / 'filter_effect_table.tex'}")
    print(f"✓ CSV:   {args.out / 'filter_effect_table.csv'}")