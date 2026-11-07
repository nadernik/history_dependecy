"""Portable path resolution for Section 4 GLM figures.

Locates the three data files this section depends on without hardcoding
absolute paths and without assuming any particular folder layout beyond
"they all sit somewhere below a common ancestor of this file".

Required files (identified by filename only):
    glm_summary_long.csv          - GLM point estimates + CIs
    glm_bootstrap_draws.parquet   - GLM bootstrap draws
    contrasts_with_axis_pattern.csv - Section-3 psychometric contrasts
                                     (one per subset; pooled/tactile/...)

Resolution algorithm:
    1. If THESIS_ROOT env variable is set and its subtree contains all
       required files, use that.
    2. Walk up from this file's location; the shallowest ancestor whose
       subtree contains all three files is the project root.
    3. Actual paths are then found by rglob under the root.

Folder layout is otherwise unconstrained: the GLM results can sit as a
sibling of the psychometric results, or be nested inside
section_4_glm/, or anywhere else in the subtree.

Output directory: figures/<subset>/ next to THIS file (not the project
root), so renaming the containing folder is harmless.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
_REQUIRED_FILES: tuple[str, ...] = (
    "glm_summary_long.csv",
    "glm_bootstrap_draws.parquet",
    "contrasts_with_axis_pattern.csv",
)
_ENV_VAR = "THESIS_ROOT"

# Cap how far up we walk when looking for the root. History_dependecy is
# typically 3-6 levels above the script; more than that risks scanning
# very large trees (e.g. the whole user home).
_MAX_ANCESTOR_LEVELS = 8

# Cap the depth of the recursive scan inside a candidate ancestor.
_MAX_SCAN_DEPTH = 10

# Directories to skip when scanning.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".idea", ".vscode",
    "site-packages",
})


# --------------------------------------------------------------------------
# Scan helper
# --------------------------------------------------------------------------
def _scan_for(root: Path, filenames: set[str],
              max_depth: int = _MAX_SCAN_DEPTH) -> dict[str, Path]:
    """BFS through `root`, up to max_depth deep. Return the first match for
    each filename in `filenames`. Missing files are simply absent from the
    result dict. Hidden and virtual-env-like directories are skipped."""
    found: dict[str, Path] = {}
    remaining = set(filenames)
    if not remaining or not root.is_dir():
        return found
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and remaining:
        current, depth = stack.pop(0)
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if not remaining:
                break
            try:
                if entry.is_file():
                    if entry.name in remaining:
                        found[entry.name] = entry
                        remaining.discard(entry.name)
                elif entry.is_dir() and depth < max_depth:
                    name = entry.name
                    if name in _SKIP_DIRS or name.startswith("."):
                        continue
                    stack.append((entry, depth + 1))
            except (PermissionError, OSError):
                continue
    return found


# --------------------------------------------------------------------------
# Root discovery
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def find_project_root(start: Path | None = None) -> Path:
    """Return the shallowest ancestor of `start` (default: this file) whose
    subtree contains all files listed in `_REQUIRED_FILES`.

    Precedence:
        1. THESIS_ROOT env variable, if valid.
        2. Walk up from `start`, checking each ancestor's subtree.
    """
    required = set(_REQUIRED_FILES)

    env = os.environ.get(_ENV_VAR)
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            found = _scan_for(p, required)
            if set(found.keys()) == required:
                return p
            missing = required - set(found.keys())
            raise FileNotFoundError(
                f"{_ENV_VAR} is set to {p}, but the following required "
                f"files were not found in its subtree: {sorted(missing)}."
            )

    if start is None:
        start = Path(__file__).resolve()

    ancestors = [start.parent] + list(start.parents)
    for candidate in ancestors[:_MAX_ANCESTOR_LEVELS + 1]:
        found = _scan_for(candidate, required)
        if set(found.keys()) == required:
            return candidate

    raise FileNotFoundError(
        f"Could not locate project root. Scanned up to "
        f"{_MAX_ANCESTOR_LEVELS} ancestors of {start}, looking for a "
        f"subtree that contains all of: {sorted(required)}. "
        f"Set {_ENV_VAR} to a directory that contains them, or check "
        f"the files are named as expected."
    )


# --------------------------------------------------------------------------
# File-specific resolvers
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _resolve(filename: str, subset: str | None = None,
             root: Path | None = None) -> Path:
    """Find `filename` under the project root (or `root` if given). If
    `subset` is given, prefer matches whose path contains a component
    equal to `subset` (case-insensitive) — this disambiguates when the
    same filename lives under multiple subset folders."""
    root = root or find_project_root()
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found under {root}"
            + (f" (subset='{subset}')" if subset else "")
        )
    if subset is not None:
        subset_lower = subset.lower()
        subset_matches = [
            m for m in matches
            if any(part.lower() == subset_lower for part in m.parts)
        ]
        if subset_matches:
            subset_matches.sort(key=lambda p: (len(p.parts), str(p)))
            return subset_matches[0]
    # Fall back to shortest path.
    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches[0]


def glm_summary_path(root: Path | None = None) -> Path:
    return _resolve("glm_summary_long.csv", root=root)


def glm_bootstrap_path(root: Path | None = None) -> Path:
    return _resolve("glm_bootstrap_draws.parquet", root=root)


def psychometric_contrasts_path(
    subset: str = "pooled",
    filename: str = "contrasts_with_axis_pattern.csv",
    root: Path | None = None,
) -> Path:
    return _resolve(filename, subset=subset, root=root)


def figures_out_dir(subset: str = "pooled", root: Path | None = None) -> Path:
    """Output directory for figures: `figures/<subset>/` next to THIS file.
    The `root` argument is accepted for backward compatibility with older
    callers but is ignored — output always lives next to the script."""
    _ = root  # explicitly ignored
    out = Path(__file__).resolve().parent / "figures" / subset
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------
# Self-test (run this file directly to verify resolution)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        root = find_project_root()
        print(f"Project root:   {root}")
        print(f"THESIS_ROOT:    {os.environ.get(_ENV_VAR, '(not set)')}")
        print()
        checks = [
            ("GLM summary",    glm_summary_path(root)),
            ("GLM bootstrap",  glm_bootstrap_path(root)),
            ("Psy (pooled)",   psychometric_contrasts_path("pooled", root=root)),
            ("Figures dir",    figures_out_dir("pooled")),
        ]
        for name, p in checks:
            status = "OK" if Path(p).exists() else "MISSING"
            print(f"  [{status:>7}] {name}: {p}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")