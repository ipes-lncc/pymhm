"""Check published documentation routes, figures, downloads and section anchors.

The generated HTML is the contract: unpublished sources need no page, while
every link in published content must resolve to a built page or asset. This
standard-library check makes no network requests and complements MathJax checks.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

_SITE_URL = "https://ipes-lncc.github.io/pymhm/"
_REPO_URL = "https://github.com/ipes-lncc/pymhm"
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_CONTENT_ATTRIBUTES = {
    "a": ("href",),
    "audio": ("src",),
    "embed": ("src",),
    "iframe": ("src",),
    "img": ("src",),
    "object": ("data",),
    "source": ("src",),
    "video": ("src", "poster"),
}


@dataclass(frozen=True)
class _Reference:
    """Locate a URL in rendered content or an optional stylesheet/script asset."""

    line: int
    url: str
    content: bool
    asset: bool


class PageLinksParser(HTMLParser):
    """Collect real HTML anchors and content links, including generated API IDs.

    Material pages identify content with ``article``, ``main`` or ``md-content``.
    Simple HTML without those wrappers is checked in full. Navigation and source
    editing controls are excluded when a content wrapper is present.
    """

    def __init__(self) -> None:
        """Initialize the page's anchor set and line-numbered URL inventory."""
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.references: list[_Reference] = []
        self.stack: list[tuple[str, bool]] = []
        self.has_content = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record anchors and resource URLs with their current content ancestry."""
        attributes = dict(attrs)
        anchor = attributes.get("id") or (attributes.get("name") if tag == "a" else None)
        if anchor:
            self.ids.add(anchor)
        content = (
            tag in {"article", "main"} or "md-content" in (attributes.get("class") or "").split()
        )
        self.has_content |= content
        content |= any(item[1] for item in self.stack)
        asset = tag in {"script", "link"}
        names = (
            ("src",)
            if tag == "script"
            else ("href",)
            if tag == "link"
            else _CONTENT_ATTRIBUTES.get(tag, ())
        )
        for name in names:
            value = (attributes.get(name) or "").strip()
            if value:
                self.references.append(_Reference(self.getpos()[0], value, content, asset))
        if tag not in _VOID_TAGS:
            self.stack.append((tag, content))

    def handle_endtag(self, tag: str) -> None:
        """Restore content ancestry even for HTML containing optional end tags."""
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return


def _untracked_asset_issues(site_dir: Path, docs_dir: Path) -> list[str]:
    """Check copied documentation assets against the real repository's Git index.

    Generated theme resources and rendered HTML normally have no counterpart in
    ``docs_dir`` and require no source entry. Excluded sources are not scanned:
    only files actually copied into the publication are candidates. Staged new
    assets count as tracked; ignored or unstaged assets cannot qualify a build.
    """
    docs_root = docs_dir.resolve()
    try:
        repository = subprocess.run(
            ["git", "-C", str(docs_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repository_root = Path(repository.stdout.strip()).resolve()
        index = subprocess.run(
            ["git", "-C", str(repository_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "Tracked asset checks require Git and a repository containing docs_dir; "
            "run this check from a repository checkout with Git available"
        ) from error
    tracked = {repository_root / name for name in index.stdout.split("\0") if name}
    issues: list[str] = []
    for published in sorted(site_dir.rglob("*")):
        source = docs_root / published.relative_to(site_dir)
        if published.is_file() and source.is_file() and source not in tracked:
            issues.append(
                f"{published}: Published documentation asset is not tracked by Git: "
                f"{source.relative_to(repository_root)}; stage the asset and update "
                ".gitignore if necessary"
            )
    return issues


def audit_links(
    site_dir: Path,
    site_url: str = _SITE_URL,
    repo_url: str = _REPO_URL,
    *,
    include_assets: bool = False,
    docs_dir: Path = Path("docs"),
    check_tracked: bool = False,
) -> tuple[int, int, list[str]]:
    """Validate published content URLs and HTML fragments against the built site.

    Absolute URLs on ``site_url`` and root-relative paths retain the configured
    deployment prefix, such as ``/pymhm/``. Queries do not change file identity;
    paths and fragments are percent-decoded. External schemes and hosts are
    ignored, except repository ``blob``/``tree`` links to documentation Markdown:
    those reports must instead link to their rendered pages. Optional asset
    checking additionally covers local stylesheet and script URLs. With
    ``check_tracked``, every published file copied from ``docs_dir`` must be in
    the Git index; generated theme files have no direct documentation source.
    """
    root = site_dir.resolve()
    base = urlsplit(site_url.rstrip("/") + "/")
    repository = urlsplit(repo_url.rstrip("/"))
    if base.scheme not in {"http", "https"} or not base.netloc:
        raise ValueError("site_url must be an absolute HTTP(S) URL")
    html_files = sorted(root.rglob("*.html"))
    if not html_files:
        raise ValueError(f"No generated HTML files found in {site_dir}")
    parsed: dict[Path, PageLinksParser] = {}
    for path in html_files:
        parser = PageLinksParser()
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
        parsed[path] = parser
    issues: list[str] = []
    checked = 0
    for path, parser in parsed.items():
        relative = path.relative_to(root).as_posix()
        page_url = urljoin(
            base.geturl(),
            relative.removesuffix("index.html") if path.name == "index.html" else relative,
        )
        for reference in parser.references:
            if reference.asset:
                if not include_assets:
                    continue
            elif parser.has_content and not reference.content:
                continue
            checked += 1
            location = f"{path}:{reference.line}"
            url = urlsplit(urljoin(page_url, reference.url))
            if url.netloc == repository.netloc and url.path.startswith(repository.path + "/"):
                suffix = unquote(url.path[len(repository.path) + 1 :])
                if (
                    suffix.startswith(("blob/", "tree/"))
                    and "/docs/" in suffix
                    and suffix.endswith(".md")
                ):
                    issues.append(
                        f"{location}: Repository Markdown replaces a rendered "
                        f"documentation page: {reference.url}"
                    )
            if url.scheme != base.scheme or url.netloc != base.netloc:
                continue
            decoded = unquote(url.path)
            if not decoded.startswith(base.path):
                issues.append(
                    f"{location}: Local URL leaves deployment prefix {base.path}: {reference.url}"
                )
                continue
            target = (root / decoded[len(base.path) :]).resolve()
            if not target.is_relative_to(root):
                issues.append(f"{location}: Local URL leaves the built site: {reference.url}")
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                issues.append(f"{location}: Missing local page or asset: {reference.url}")
                continue
            fragment = unquote(url.fragment).split(":~:text=", 1)[0]
            if fragment and target in parsed and fragment not in parsed[target].ids:
                issues.append(
                    f"{location}: Missing HTML anchor #{fragment} in "
                    f"{target.relative_to(root)}: {reference.url}"
                )
    if check_tracked:
        issues.extend(_untracked_asset_issues(root, docs_dir))
    return len(parsed), checked, issues


def main() -> None:
    """Fail the documentation gate when published content points to missing results."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--site-dir", type=Path, default=Path("site"))
    arguments.add_argument("--site-url", default=_SITE_URL)
    arguments.add_argument("--repo-url", default=_REPO_URL)
    arguments.add_argument("--include-assets", action="store_true")
    arguments.add_argument("--docs-dir", type=Path, default=Path("docs"))
    arguments.add_argument("--check-tracked", action="store_true")
    args = arguments.parse_args()
    try:
        pages, links, issues = audit_links(
            args.site_dir,
            args.site_url,
            args.repo_url,
            include_assets=args.include_assets,
            docs_dir=args.docs_dir,
            check_tracked=args.check_tracked,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if issues:
        raise SystemExit("Documentation link validation failed:\n" + "\n".join(issues))
    print(f"Validated documentation links: {pages} HTML pages, {links} URLs")


if __name__ == "__main__":
    main()
