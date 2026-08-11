#!/usr/bin/env python3

"""
Recursively print the Python import tree for one or more source files.

This uses Python's AST rather than actually importing modules, so it can
discover dependencies even when some third-party packages are not installed.

Example:

    python tools/print_import_tree.py \
        model_training_and_implementation/src/handoff_detection_wrapper.py \
        model_training_and_implementation/src/tabm_handoff_detection_wrapper.py

You can also specify the repository root explicitly:

    python tools/print_import_tree.py \
        --root . \
        model_training_and_implementation/src/handoff_detection_wrapper.py
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


STANDARD_LIBRARY = set(getattr(sys, "stdlib_module_names", ()))


def parse_imports(file_path: Path) -> List[Tuple[str, int, Optional[str]]]:
    """
    Return imports found in a Python file.

    Each result is:
        (module_name, relative_level, imported_name)

    Examples:

        import numpy
            -> ("numpy", 0, None)

        from pathlib import Path
            -> ("pathlib", 0, "Path")

        from .foo import Bar
            -> ("foo", 1, "Bar")

        from ..utils.foo import Bar
            -> ("utils.foo", 2, "Bar")
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except Exception as exc:
        print(
            f"[WARNING] Could not parse {file_path}: {exc}",
            file=sys.stderr,
        )
        return []

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    (
                        alias.name,
                        0,
                        None,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""

            for alias in node.names:
                imports.append(
                    (
                        module_name,
                        node.level,
                        alias.name,
                    )
                )

    return imports


def package_parts_for_file(
    file_path: Path,
    repo_root: Path,
) -> List[str]:
    """
    Determine package components containing file_path.

    Example:

        repo/model_training_and_implementation/src/foo.py

    becomes:

        ["model_training_and_implementation", "src"]

    assuming those directories are Python packages.
    """
    parent = file_path.parent.resolve()

    parts = []

    while parent != repo_root.parent:
        if not (parent / "__init__.py").exists():
            break

        parts.insert(0, parent.name)

        if parent == repo_root:
            break

        parent = parent.parent

    return parts


def resolve_absolute_local_module(
    module_name: str,
    repo_root: Path,
) -> Optional[Path]:
    """
    Resolve an absolute import to a Python file inside repo_root.
    """
    parts = module_name.split(".") if module_name else []

    if not parts:
        return None

    module_file = repo_root.joinpath(*parts).with_suffix(".py")

    if module_file.is_file():
        return module_file.resolve()

    package_init = repo_root.joinpath(*parts, "__init__.py")

    if package_init.is_file():
        return package_init.resolve()

    return None


def resolve_relative_local_module(
    current_file: Path,
    module_name: str,
    relative_level: int,
) -> Optional[Path]:
    """
    Resolve imports such as:

        from .foo import Bar
        from ..utils.foo import Bar
    """
    base = current_file.parent.resolve()

    # level=1 means current package.
    # level=2 means parent package, etc.
    for _ in range(max(0, relative_level - 1)):
        base = base.parent

    parts = module_name.split(".") if module_name else []

    candidate = base.joinpath(*parts)

    module_file = candidate.with_suffix(".py")

    if module_file.is_file():
        return module_file.resolve()

    package_init = candidate / "__init__.py"

    if package_init.is_file():
        return package_init.resolve()

    return None


def resolve_imported_symbol_as_module(
    base_module_path: Path,
    imported_name: Optional[str],
) -> Optional[Path]:
    """
    Handle cases like:

        from .foo import bar

    where `bar` may itself be the module:

        foo/bar.py
    """
    if not imported_name:
        return None

    if base_module_path.name == "__init__.py":
        package_dir = base_module_path.parent
    else:
        package_dir = base_module_path.parent

    candidate = package_dir / f"{imported_name}.py"

    if candidate.is_file():
        return candidate.resolve()

    package_candidate = package_dir / imported_name / "__init__.py"

    if package_candidate.is_file():
        return package_candidate.resolve()

    return None


def classify_external(module_name: str) -> str:
    """
    Classify an external module as stdlib or third-party.
    """
    top_level = module_name.split(".")[0]

    if top_level in STANDARD_LIBRARY:
        return "stdlib"

    return "third-party"


def print_tree(
    file_path: Path,
    repo_root: Path,
    visited_stack: Set[Path],
    external_modules: Dict[str, Set[str]],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    """
    Recursively print the local import tree.
    """
    branch = "└── " if is_last else "├── "

    try:
        display_path = file_path.relative_to(repo_root)
    except ValueError:
        display_path = file_path

    print(f"{prefix}{branch}{display_path}")

    if file_path in visited_stack:
        child_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{child_prefix}└── [already in current import chain]")
        return

    visited_stack = set(visited_stack)
    visited_stack.add(file_path)

    imports = parse_imports(file_path)

    children = []
    seen_children = set()

    for module_name, relative_level, imported_name in imports:

        local_path = None

        if relative_level > 0:
            local_path = resolve_relative_local_module(
                current_file=file_path,
                module_name=module_name,
                relative_level=relative_level,
            )

        else:
            local_path = resolve_absolute_local_module(
                module_name=module_name,
                repo_root=repo_root,
            )

        if local_path is not None:
            if local_path not in seen_children:
                children.append(
                    (
                        "local",
                        local_path,
                        module_name,
                    )
                )
                seen_children.add(local_path)

            # Check whether:
            #
            # from package import submodule
            #
            # refers to another local Python module.
            symbol_module = resolve_imported_symbol_as_module(
                base_module_path=local_path,
                imported_name=imported_name,
            )

            if (
                symbol_module is not None
                and symbol_module not in seen_children
            ):
                children.append(
                    (
                        "local",
                        symbol_module,
                        imported_name,
                    )
                )
                seen_children.add(symbol_module)

            continue

        # Relative import that didn't resolve is likely local but broken.
        if relative_level > 0:
            import_string = "." * relative_level + module_name

            if imported_name:
                import_string += f" import {imported_name}"

            children.append(
                (
                    "unresolved",
                    None,
                    import_string,
                )
            )

            continue

        if not module_name:
            continue

        top_level = module_name.split(".")[0]

        category = classify_external(module_name)

        external_modules.setdefault(category, set()).add(top_level)

        children.append(
            (
                category,
                None,
                module_name,
            )
        )

    child_prefix = prefix + ("    " if is_last else "│   ")

    for index, child in enumerate(children):
        child_is_last = index == len(children) - 1

        child_type, child_path, child_name = child

        if child_type == "local":
            print_tree(
                file_path=child_path,
                repo_root=repo_root,
                visited_stack=visited_stack,
                external_modules=external_modules,
                prefix=child_prefix,
                is_last=child_is_last,
            )

        else:
            child_branch = (
                "└── "
                if child_is_last
                else "├── "
            )

            if child_type == "stdlib":
                label = "[stdlib]"
            elif child_type == "third-party":
                label = "[third-party]"
            else:
                label = "[unresolved]"

            print(
                f"{child_prefix}"
                f"{child_branch}"
                f"{child_name} "
                f"{label}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the recursive import tree for Python files "
            "without importing them."
        )
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Python entry-point file(s) to analyze.",
    )

    parser.add_argument(
        "--root",
        default=".",
        help=(
            "Repository root used to resolve local imports. "
            "Default: current directory."
        ),
    )

    args = parser.parse_args()

    repo_root = Path(args.root).resolve()

    external_modules: Dict[str, Set[str]] = {
        "stdlib": set(),
        "third-party": set(),
    }

    print()
    print("=" * 80)
    print("IMPORT TREE")
    print("=" * 80)

    for file_name in args.files:
        file_path = Path(file_name)

        if not file_path.is_absolute():
            file_path = (
                Path.cwd()
                / file_path
            )

        file_path = file_path.resolve()

        if not file_path.is_file():
            print(
                f"\n[ERROR] File not found: {file_path}",
                file=sys.stderr,
            )
            continue

        print()

        print_tree(
            file_path=file_path,
            repo_root=repo_root,
            visited_stack=set(),
            external_modules=external_modules,
        )

    print()
    print("=" * 80)
    print("THIRD-PARTY TOP-LEVEL IMPORTS")
    print("=" * 80)

    for module in sorted(
        external_modules["third-party"]
    ):
        print(module)

    print()
    print("=" * 80)
    print("STANDARD-LIBRARY IMPORTS")
    print("=" * 80)

    for module in sorted(
        external_modules["stdlib"]
    ):
        print(module)

    print()


if __name__ == "__main__":
    main()