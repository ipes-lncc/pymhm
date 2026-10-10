"""Published reports keep working figures, downloads and rendered section links."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_docs_links as checker


def site_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write a small built site with actual route, asset and anchor targets."""
    site = tmp_path / "site"
    site.mkdir()
    for name, content in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return site


def test_maxwell_report_cannot_fall_back_to_github_or_missing_figures(tmp_path: Path) -> None:
    """Catch the reported Gallery navigation defect and the missing result assets."""
    site = site_tree(
        tmp_path,
        {
            "index.html": "<main><a href='gallery/maxwell/'>Device</a></main>",
            "gallery/maxwell/index.html": (
                "<article><a href='https://github.com/ipes-lncc/pymhm/blob/main/docs/"
                "cases/maxwell-nanoguide.md#fields'>Complete result report</a>"
                "<img src='../../figures/maxwell/components.png'>"
                "<a href='../../figures/maxwell/comparison.json'>Field record</a></article>"
            ),
        },
    )
    pages, urls, issues = checker.audit_links(site)
    assert pages == 2 and urls == 4
    assert len(issues) == 3
    assert any(
        "Repository Markdown" in issue and "maxwell-nanoguide.md" in issue for issue in issues
    )
    assert any("Missing local" in issue and "components.png" in issue for issue in issues)
    assert any("Missing local" in issue and "comparison.json" in issue for issue in issues)


def test_rendered_routes_fragments_and_percent_encoded_assets(tmp_path: Path) -> None:
    """Accept pretty and flat routes, deployment prefixes and generated API IDs."""
    site = site_tree(
        tmp_path,
        {
            "index.html": (
                "<main><a href='/pymhm/cases/maxwell/#fields'>Results</a>"
                "<a href='https://ipes-lncc.github.io/pymhm/api.html#pymhm.core.LocalEquations'>API</a>"
                "<a href='api.html#old%20anchor'>Named anchor</a>"
                "<a href='api.html#:~:text=Equations'>Text fragment</a>"
                "<a href='api.html#pymhm.core.LocalEquations:~:text=Equations'>Anchor and text</a>"
                "<a href='api.html?view=compact'>Query</a></main>"
            ),
            "cases/maxwell/index.html": (
                "<div class='md-content'><h2 id='fields'>Fields</h2>"
                "<img src='../../figures/components%20%CE%B1.png?download=1'>"
                "<a href='/pymhm/records/field.json'>Norms</a>"
                "<a href='../../figures/curve.svg#layer'>Vector plot</a>"
                "<a href='#fields'>This section</a><a href='../../'>Home</a></div>"
            ),
            "api.html": (
                "<h2 id='pymhm.core.LocalEquations'>Equations</h2><a name='old anchor'></a>"
            ),
            "figures/components α.png": "image",
            "figures/curve.svg": "<svg></svg>",
            "records/field.json": "{}",
        },
    )
    assert checker.audit_links(site) == (3, 11, [])


@pytest.mark.parametrize(
    "href,message",
    [
        ("cases/missing/", "Missing local page"),
        ("#absent", "Missing HTML anchor #absent"),
        ("/assets/image.png", "leaves deployment prefix"),
        ("../outside.html", "leaves deployment prefix"),
        ("%2e%2e/outside.html", "leaves the built site"),
    ],
)
def test_missing_routes_anchors_and_deployment_escapes(
    tmp_path: Path, href: str, message: str
) -> None:
    """Every published local reference resolves inside the deployed site."""
    site = site_tree(tmp_path, {"index.html": f"<main><a href='{href}'>Result</a></main>"})
    issues = checker.audit_links(site)[2]
    assert len(issues) == 1 and message in issues[0]


def test_external_urls_and_repository_source_links_are_intentional(tmp_path: Path) -> None:
    """Ignore external schemes but reject both blob and tree Markdown fallbacks."""
    site = site_tree(
        tmp_path,
        {
            "index.html": (
                "<article><a href='https://example.org/results/'>External</a>"
                "<a href='//example.org/results/'>Protocol relative</a>"
                "<a href='mailto:group@example.org'>Email</a>"
                "<a href='javascript:void(0)'>Action</a>"
                "<img src='data:image/png;base64,AAAA'>"
                "<a href='https://github.com/another/project/blob/main/docs/result.md'>"
                "Other project</a>"
                "<a href='https://github.com/ipes-lncc/pymhm/blob/main/notebooks/case.ipynb'>Notebook</a>"
                "<a href='https://github.com/ipes-lncc/pymhm/tree/main/docs/'>Sources</a>"
                "<a href='https://github.com/ipes-lncc/pymhm/edit/main/docs/index.md'>"
                "Edit source</a>"
                "<a href='https://github.com/ipes-lncc/pymhm/tree/feature/topic/docs/cases/result.md'>Report</a>"
                "<a href=''></a><img alt='No source'></article>"
            )
        },
    )
    pages, urls, issues = checker.audit_links(site)
    assert pages == 1 and urls == 10
    assert len(issues) == 1 and "Repository Markdown" in issues[0]


def test_content_scope_media_and_optional_script_assets(tmp_path: Path) -> None:
    """Ignore navigation controls, check displayed media and optionally CSS/JS."""
    site = site_tree(
        tmp_path,
        {
            "index.html": (
                "<head><link href='assets/theme.css'><script src='assets/view.js'></script></head>"
                "<nav><a href='not-a-content-route/'>Navigation</a></nav>"
                "<main><article><video src='assets/movie.mp4' poster='assets/poster.png'></video>"
                "<audio src='assets/audio.ogg'></audio><source src='assets/sample.webm'>"
                "<iframe src='frame.html'></iframe><object data='assets/chart.svg'></object>"
                "<embed src='assets/chart.svg'></unknown></article></main></unknown>"
            ),
            "frame.html": "<p>Frame</p>",
            "assets/movie.mp4": "movie",
            "assets/poster.png": "poster",
            "assets/audio.ogg": "audio",
            "assets/sample.webm": "sample",
            "assets/chart.svg": "chart",
        },
    )
    assert checker.audit_links(site) == (2, 7, [])
    issues = checker.audit_links(site, include_assets=True)[2]
    assert len(issues) == 2 and all("Missing local" in issue for issue in issues)
    (site / "assets/theme.css").write_text("", encoding="utf-8")
    (site / "assets/view.js").write_text("", encoding="utf-8")
    assert checker.audit_links(site, include_assets=True) == (2, 9, [])


def test_checker_cli_success_and_explicit_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the real script entry point and readable failure diagnostics."""
    site = site_tree(tmp_path, {"index.html": "<p>Plain home</p>"})
    monkeypatch.setattr(sys, "argv", ["check_docs_links", "--site-dir", str(site)])
    runpy.run_path(str(Path(checker.__file__)), run_name="__main__")
    assert "1 HTML pages, 0 URLs" in capsys.readouterr().out
    monkeypatch.setattr(
        sys, "argv", ["check_docs_links", "--site-dir", str(site), "--site-url", "relative/"]
    )
    with pytest.raises(SystemExit, match="absolute HTTP"):
        checker.main()
    (site / "index.html").write_text("<a href='missing.json'>Record</a>", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["check_docs_links", "--site-dir", str(site), "--include-assets"]
    )
    with pytest.raises(SystemExit, match="Documentation link validation failed"):
        checker.main()
    (site / "index.html").unlink()
    with pytest.raises(ValueError, match="No generated HTML"):
        checker.audit_links(site)


def test_published_assets_must_be_staged_even_when_present_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reproduce an ignored PNG passing locally but missing from a clean checkout."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("docs/figures/*.png\n", encoding="utf-8")
    docs = tmp_path / "docs"
    (docs / "figures").mkdir(parents=True)
    image = docs / "figures/material.png"
    image.write_text("image", encoding="utf-8")
    record = docs / "figures/record.json"
    record.write_text("{}", encoding="utf-8")
    site = site_tree(
        tmp_path,
        {
            "index.html": (
                "<script src='assets/generated.js'></script><main>"
                "<img src='figures/material.png'><a href='figures/record.json'>Norms</a></main>"
            ),
            "figures/material.png": "image",
            "figures/record.json": "{}",
            "assets/generated.js": "generated theme resource",
        },
    )
    assert checker.audit_links(site, include_assets=True) == (1, 3, [])
    issues = checker.audit_links(site, docs_dir=docs, check_tracked=True)[2]
    assert len(issues) == 2 and all("not tracked by Git" in issue for issue in issues)
    assert any("material.png" in issue for issue in issues)
    assert any("record.json" in issue for issue in issues)
    subprocess.run(["git", "-C", str(tmp_path), "add", str(record)], check=True)
    assert len(checker.audit_links(site, docs_dir=docs, check_tracked=True)[2]) == 1
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", str(image)], check=True)
    (docs / "figures/unpublished.png").write_text("unpublished", encoding="utf-8")
    assert checker.audit_links(site, docs_dir=docs, check_tracked=True) == (1, 2, [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_docs_links",
            "--site-dir",
            str(site),
            "--docs-dir",
            str(docs),
            "--check-tracked",
            "--include-assets",
        ],
    )
    checker.main()
    assert "1 HTML pages, 3 URLs" in capsys.readouterr().out


def test_tracked_check_requires_an_actual_checkout_and_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional Git contract fails clearly outside a checkout or without Git."""
    docs = tmp_path / "docs"
    docs.mkdir()
    site = site_tree(tmp_path, {"index.html": "<p>Home</p>"})
    assert checker.audit_links(site, docs_dir=docs) == (1, 0, [])
    with pytest.raises(ValueError, match="require Git and a repository"):
        checker.audit_links(site, docs_dir=docs, check_tracked=True)

    def unavailable_git(*args: object, **kwargs: object) -> None:
        """Model the explicit missing-executable contract without changing PATH."""
        raise FileNotFoundError("git")

    monkeypatch.setattr(checker.subprocess, "run", unavailable_git)
    with pytest.raises(ValueError, match="Git available"):
        checker.audit_links(site, docs_dir=docs, check_tracked=True)
