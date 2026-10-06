"""Publish introductory notebooks as static Markdown with their executed figures.

Run in the locked ``introduction`` Pixi environment after ``notebooks-run``.
Documentation builds consume the published pages without executing notebooks or
importing an optional finite-element backend. Source and execution digests retain
the provenance of every published numerical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import nbformat
from nbconvert import MarkdownExporter
from nbformat import NotebookNode

_SOURCE_URL = "https://github.com/ipes-lncc/pymhm/blob/main/notebooks/introduction"
_DISPLAY = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+[^`]*`+)")
_LINK = re.compile(r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<url><[^>\n]+>|[^\s)]+)")
_REFERENCE_LINK = re.compile(r"(?m)^(?P<prefix>[ \t]*\[[^\]\n]+\]:[ \t]*)(?P<url><[^>\n]+>|[^\s]+)")
_TEMPLATE = """{% extends 'index.md.j2' %}
{% macro figure(output, mime) %}
{% set path = output.metadata.filenames[mime] | path2url %}
[![{{ output.metadata.tutorial_alt }}]({{ path }})]({{ path }})
{% endmacro %}
{% block data_png %}{{ figure(output, 'image/png') }}{% endblock data_png %}
{% block data_svg %}{{ figure(output, 'image/svg+xml') }}{% endblock data_svg %}
{% block data_jpg %}{{ figure(output, 'image/jpeg') }}{% endblock data_jpg %}
{% block stream %}
{{ output.text | strip_ansi | tutorial_output }}
{% endblock stream %}
{% block data_text scoped %}
{{ output.data['text/plain'] | strip_ansi | tutorial_output }}
{% endblock data_text %}
"""


@dataclass(frozen=True)
class _RenderedTutorial:
    """Keep one validated page, its extracted assets and its provenance together."""

    page: Path
    markdown: str
    assets: dict[Path, bytes]
    provenance: dict[str, Any]


def _format_text_output(text: str, *, root: Path | None = None) -> str:
    """Keep numerical output verbatim and collapse long provenance records.

    Code fences protect printed TeX and Markdown-like metadata from rendering.
    Checkout paths are displayed relative to the repository root. Numerical data
    are unchanged, and the execution digest always refers to the original file.
    """
    text = text.strip("\n")
    if root is not None:
        text = text.replace(str(root), ".")
    longest = max((len(match.group()) for match in re.finditer(r"`{3,}", text)), default=2)
    fence = "`" * (longest + 1)
    block = fence + "text\n" + text + "\n" + fence
    if len(text) <= 5000 and len(text.splitlines()) <= 40:
        return block
    return '??? note "Numerical output and provenance"\n\n' + "\n".join(
        "    " + line for line in block.splitlines()
    )


def _map_markdown_prose(markdown: str, transform: Callable[[str], str]) -> str:
    """Transform Markdown prose while preserving fenced code blocks verbatim."""
    result: list[str] = []
    prose: list[str] = []
    fence = ""

    def flush_prose() -> None:
        """Transform a prose segment before a code fence or the end of a cell."""
        if prose:
            result.append(transform("".join(prose)))
            prose.clear()

    for line in markdown.splitlines(keepends=True):
        match = _FENCE.match(line)
        if match:
            delimiter = match.group(1)
            if not fence:
                flush_prose()
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = ""
            result.append(line)
        elif fence:
            result.append(line)
        else:
            prose.append(line)
    flush_prose()
    return "".join(result)


def _normalize_display_math(markdown: str) -> str:
    """Put display delimiters on separate lines without modifying fenced code.

    Inline mathematics and the TeX inside a display retain their original text.
    Standalone delimiters and blank lines let MkDocs preserve notebook displays.
    """
    return _map_markdown_prose(
        markdown,
        lambda text: _DISPLAY.sub(
            lambda match: "\n\n$$\n" + match.group(1).strip() + "\n$$\n\n", text
        ),
    )


def _rebase_markdown_links(markdown: str, *, source: Path, root: Path, page: Path) -> str:
    """Resolve notebook-relative prose links for their published documentation page.

    Documentation targets remain local links. Other repository targets use their
    public GitHub source URL. External links, anchors, queries and code stay intact.
    """

    def replace_link(match: re.Match[str]) -> str:
        """Rebase one inline link destination, preserving its label and title."""
        destination = match.group("url")
        angled = destination.startswith("<") and destination.endswith(">")
        parsed = urlsplit(destination[1:-1] if angled else destination)
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
            return match.group()
        target = (source.parent / unquote(parsed.path)).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Notebook link leaves the repository: {destination}")
        if target.is_relative_to(root / "docs"):
            path = Path(os.path.relpath(target, root / "docs" / page.parent)).as_posix()
            url = urlunsplit(("", "", quote(path, safe="/"), parsed.query, parsed.fragment))
        else:
            path = "/ipes-lncc/pymhm/blob/main/" + target.relative_to(root).as_posix()
            url = urlunsplit(
                ("https", "github.com", quote(path, safe="/"), parsed.query, parsed.fragment)
            )
        return match.group("prefix") + ("<" + url + ">" if angled else url)

    def rebase_prose(text: str) -> str:
        """Keep inline code intact while rebasing destinations in surrounding prose."""
        parts = _INLINE_CODE.split(text)
        return "".join(
            part if index % 2 else _REFERENCE_LINK.sub(replace_link, _LINK.sub(replace_link, part))
            for index, part in enumerate(parts)
        )

    return _map_markdown_prose(markdown, rebase_prose)


def _validated_execution(source: Path, executed: Path) -> NotebookNode:
    """Require an execution of the exact source cells, with plots and no errors.

    Execution counts establish that every nonempty code cell ran. They are not a
    numerical acceptance test; the notebooks own their physical and error checks.
    """
    if not executed.is_file():
        raise ValueError(f"Missing executed notebook: {executed}")
    original = nbformat.read(source, as_version=4)
    result = nbformat.read(executed, as_version=4)
    if len(original.cells) != len(result.cells):
        raise ValueError(f"Source/execution cell counts differ: {source.name}")
    plot_count = 0
    supported = {"image/svg+xml", "image/png", "image/jpeg", "text/markdown", "text/plain"}
    for index, (before, after) in enumerate(zip(original.cells, result.cells, strict=True), 1):
        if before.cell_type != after.cell_type or before.source != after.source:
            raise ValueError(f"Stale execution: {source.name}, cell {index}")
        if after.cell_type != "code":
            continue
        if after.source.strip() and not isinstance(after.get("execution_count"), int):
            raise ValueError(f"Unexecuted code: {source.name}, cell {index}")
        for output in after.get("outputs", []):
            if output.output_type == "error":
                raise ValueError(f"Error output: {source.name}, cell {index}")
            data = output.get("data", {})
            if data and not supported.intersection(data):
                raise ValueError(f"No static output representation: {source.name}, cell {index}")
            plot_count += bool({"image/svg+xml", "image/png", "image/jpeg"}.intersection(data))
    if not plot_count:
        raise ValueError(f"No executed field or convergence plots: {source.name}")
    return result


def _render_tutorial(source: Path, executed: Path, root: Path) -> _RenderedTutorial:
    """Export one verified execution while preserving its code and output order."""
    notebook = deepcopy(_validated_execution(source, executed))
    page = Path("tutorials/introduction") / f"{source.stem}.md"
    title = next(
        cell.source.splitlines()[0] for cell in notebook.cells if cell.cell_type == "markdown"
    )
    title = title.lstrip("# ").replace("[", r"\[").replace("]", r"\]")
    plot_number = 0
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = _rebase_markdown_links(
                _normalize_display_math(cell.source), source=source, root=root, page=page
            )
        for output in cell.get("outputs", []):
            if {"image/png", "image/svg+xml", "image/jpeg"}.intersection(output.get("data", {})):
                plot_number += 1
                output.metadata["tutorial_alt"] = f"Figure {plot_number} — {title}"
            if "text/markdown" in output.get("data", {}):
                output.data["text/markdown"] = _rebase_markdown_links(
                    _normalize_display_math(output.data["text/markdown"]),
                    source=source,
                    root=root,
                    page=page,
                )
    exporter = MarkdownExporter(raw_template=_TEMPLATE)
    exporter.register_filter("tutorial_output", partial(_format_text_output, root=root))
    exporter.display_data_priority = [
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/plain",
    ]
    image_directory = Path("assets/tutorials") / source.stem
    image_prefix = "../../" + image_directory.as_posix()
    markdown, resources = exporter.from_notebook_node(
        notebook,
        resources={"unique_key": "figure", "output_files_dir": image_prefix},
    )
    heading, _, body = markdown.partition("\n")
    markdown = (
        heading + "\n\nFollow the numbered steps: state the variational problem, choose the local "
        "and trace spaces, declare the local and global equations, solve, and inspect "
        "the physical fields.\n\n" + body.lstrip("\n")
    )
    assets: dict[Path, bytes] = {}
    for name, content in resources.get("outputs", {}).items():
        path = Path(name)
        if path.parent.as_posix() != image_prefix or path.suffix not in {
            ".png",
            ".svg",
            ".jpg",
            ".jpeg",
        }:
            raise ValueError(f"Unexpected extracted asset path: {name}")
        assets[image_directory / path.name] = content
    if not assets:
        raise ValueError(f"Exporter produced no static figures: {source.name}")
    reproduction = (
        "\n\n## Reproduce this tutorial\n\n"
        f"[View the source notebook]({_SOURCE_URL}/{source.name}) or "
        "[download the notebook](https://raw.githubusercontent.com/ipes-lncc/pymhm/"
        f"main/notebooks/introduction/{source.name}). "
        "Run its cells interactively, or execute the notebook from the repository root "
        "with the checked-in Pixi lockfile:\n\n"
        "```bash\n"
        "pixi install --locked -e introduction\n"
        f"pixi run --locked -e introduction notebooks-run introduction/{source.name} "
        "--timeout 3600\n"
        "```\n\n"
        "The runner writes the executed copy to `build/notebooks/introduction/`. "
        "The figures and numerical outputs on this page come from that execution. "
        "Timings describe the recorded hardware and solver settings; rerun performance "
        "examples on an idle machine to measure your own environment.\n"
    )
    markdown = markdown.rstrip() + reproduction
    provenance = {
        "source_notebook": source.relative_to(root).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "executed_notebook": executed.relative_to(root).as_posix()
        if executed.is_relative_to(root)
        else executed.name,
        "executed_sha256": hashlib.sha256(executed.read_bytes()).hexdigest(),
        "executed_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "page": page.as_posix(),
        "page_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "assets": [
            {"path": path.as_posix(), "sha256": hashlib.sha256(content).hexdigest()}
            for path, content in sorted(assets.items())
        ],
    }
    return _RenderedTutorial(page, markdown, assets, provenance)


def _publish(tutorials: list[_RenderedTutorial], output_dir: Path, *, check: bool) -> None:
    """Write validated pages and assets, or check their byte-for-byte freshness."""
    files: dict[Path, bytes] = {}
    for tutorial in tutorials:
        files[tutorial.page] = tutorial.markdown.encode("utf-8")
        files.update(tutorial.assets)
    manifest = {
        "schema_version": 1,
        "generator": "scripts/render_tutorials.py",
        "notebooks": [tutorial.provenance for tutorial in tutorials],
    }
    files[Path("tutorials/introduction/manifest.json")] = (
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if check:
        stale = [
            path.as_posix()
            for path, content in files.items()
            if not (output_dir / path).is_file() or (output_dir / path).read_bytes() != content
        ]
        if stale:
            raise ValueError("Published tutorial files need regeneration: " + ", ".join(stale))
        return
    for path, content in files.items():
        destination = output_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def main() -> None:
    """Render all introductory tutorials using existing verified notebook outputs."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=root / "build/notebooks/introduction",
        help="Directory containing executed copies of the source notebooks",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "docs", help="Static documentation root"
    )
    parser.add_argument("--check", action="store_true", help="Check without writing any files")
    args = parser.parse_args()
    sources = sorted((root / "notebooks/introduction").glob("*.ipynb"))
    if not sources:
        parser.error("No introductory source notebooks found")
    try:
        tutorials = [
            _render_tutorial(source, args.executed_dir.resolve() / source.name, root)
            for source in sources
        ]
        _publish(tutorials, args.output_dir, check=args.check)
    except ValueError as error:
        parser.error(str(error))
    verb = "Verified" if args.check else "Rendered"
    print(f"{verb} {len(tutorials)} tutorials and {sum(len(t.assets) for t in tutorials)} figures")


if __name__ == "__main__":
    main()
