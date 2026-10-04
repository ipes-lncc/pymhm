"""Check documentation and type contracts throughout the runtime source tree."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterator
from pathlib import Path

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _definitions(node: ast.AST, prefix: str = "") -> Iterator[tuple[str, Definition]]:
    """Yield all named definitions, including private and nested implementation helpers."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}.{child.name}" if prefix else child.name
            yield name, child
            yield from _definitions(child, name)
        else:
            yield from _definitions(child, prefix)


def _overload(definition: Definition) -> bool:
    """Identify typing-only signatures whose callable implementation owns their docs."""
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "overload")
        or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
        for decorator in definition.decorator_list
    )


def source_contract_issues(path: Path) -> tuple[list[str], int]:
    """Return deterministic diagnostics and the number of inspected source definitions.

    Every module, class and named function needs a nonempty docstring. Every
    function argument except its conventional ``self`` or ``cls`` receiver needs
    an annotation, as does its return value. This includes asynchronous functions,
    variadic arguments and nested implementation helpers. Overload signatures
    share the docstring of their documented callable implementation; their
    argument and return annotations are checked independently.
    """
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    issues = []
    if not ast.get_docstring(module):
        issues.append(f"{path}:1: module has no docstring")
    definitions = list(_definitions(module))
    documented = {
        name
        for name, definition in definitions
        if ast.get_docstring(definition) and not _overload(definition)
    }
    for name, definition in definitions:
        location = f"{path}:{definition.lineno}: {name}"
        if not ast.get_docstring(definition) and not (_overload(definition) and name in documented):
            issues.append(f"{location} has no docstring")
        if isinstance(definition, ast.ClassDef):
            continue
        arguments = definition.args
        positional = [*arguments.posonlyargs, *arguments.args]
        received = (
            positional[0].arg if positional and positional[0].arg in {"self", "cls"} else None
        )
        all_arguments = [*positional, *arguments.kwonlyargs]
        if arguments.vararg is not None:
            all_arguments.append(arguments.vararg)
        if arguments.kwarg is not None:
            all_arguments.append(arguments.kwarg)
        for argument in all_arguments:
            if argument.arg != received and argument.annotation is None:
                issues.append(f"{location} argument {argument.arg!r} has no annotation")
        if definition.returns is None:
            issues.append(f"{location} has no return annotation")
    return issues, len(definitions)


def main() -> None:
    """Audit recursively without importing optional backends or executing numerical code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/pymhm",
        help="Runtime source directory (default: src/pymhm)",
    )
    source = parser.parse_args().source
    modules = sorted(source.rglob("*.py"))
    if not modules:
        raise SystemExit(f"No Python modules found in {source}")
    issues = []
    definitions = 0
    for module in modules:
        problems, count = source_contract_issues(module)
        issues.extend(problems)
        definitions += count
    if issues:
        raise SystemExit("\n".join(issues))
    print(
        f"Validated docstrings and annotations in {len(modules)} modules "
        f"and {definitions} definitions"
    )


if __name__ == "__main__":
    main()
