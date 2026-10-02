"""Publication profiles may omit pages, never source or generated math validation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def api():
    """Load the standard-library documentation checker with its real HTML parser."""
    path = Path(__file__).resolve().parents[1] / "scripts/check_docs_math.py"
    name = "check_docs_math_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


def tree(tmp_path, supplementary="Plain source.", extra_html=None):
    """Create a canonical display and a supplementary source with optional HTML."""
    docs = tmp_path / "docs"
    site = tmp_path / "site"
    docs.mkdir()
    site.mkdir()
    (docs / "index.md").write_text("# Canonical\n\n$$\nx^2\n$$\n")
    (site / "index.html").write_text('<div class="arithmatex">\\[x^2\\]</div>')
    (docs / "supplementary.md").write_text(supplementary)
    if extra_html is not None:
        (site / "supplementary.html").write_text(extra_html)
    (docs / "publication-core.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "canonical_pages": ["index.md"],
                "excluded_pages": ["supplementary.md"],
            }
        )
    )
    return docs, site


def test_legitimate_exclusion_waives_only_missing_generated_page(api, tmp_path):
    """A supplementary source need not be compiled in the selected publication profile."""
    docs, site = tree(tmp_path)
    excluded = api.excluded_markdown_sources(docs)
    assert excluded == {Path("supplementary.md")}
    assert api.audit_site(site, docs, excluded) == (1, 1, [])
    assert any("Generated HTML page is missing" in issue for issue in api.audit_site(site, docs)[2])


def test_nonexcluded_missing_html_remains_an_error(api, tmp_path):
    """Omitting an unrelated page cannot silently broaden the manifest's exclusion."""
    docs, site = tree(tmp_path)
    (docs / "required.md").write_text("Required source.\n")
    issues = api.audit_site(site, docs, api.excluded_markdown_sources(docs))[2]
    assert any(
        "required.md" in issue and "Generated HTML page is missing" in issue for issue in issues
    )
    assert not any("supplementary.md" in issue for issue in issues)


def test_excluded_source_delimiters_are_still_checked(api, tmp_path):
    """A source excluded from compilation cannot hide an unfinished display."""
    docs, site = tree(tmp_path, supplementary="$$\nx^2\n")
    issues = api.audit_site(site, docs, api.excluded_markdown_sources(docs))[2]
    assert any(
        "supplementary.md" in issue and "Unclosed mathematical display" in issue for issue in issues
    )


def test_extra_html_is_checked_even_without_markdown_source(api, tmp_path):
    """Every generated page is parsed, including files not named by the profile."""
    docs, site = tree(tmp_path)
    (site / "extra.html").write_text("<p>Unprotected \\frac{1}{2}</p>")
    pages, expressions, issues = api.audit_site(site, docs, api.excluded_markdown_sources(docs))
    assert pages == 2
    assert expressions == 1
    assert any("extra.html" in issue and "Unprotected TeX" in issue for issue in issues)


def test_generated_excluded_html_does_not_bypass_conversion_checks(api, tmp_path):
    """Exclusion does not waive display counts when that page was actually generated."""
    docs, site = tree(tmp_path, supplementary="$$\nx^2\n$$\n", extra_html="<p>Lost display</p>")
    issues = api.audit_site(site, docs, api.excluded_markdown_sources(docs))[2]
    assert any(
        "supplementary.md" in issue and "1 Markdown display blocks" in issue for issue in issues
    )
    (site / "supplementary.html").write_text('<div class="arithmatex">\\[x^2\\]</div>')
    assert api.audit_site(site, docs, api.excluded_markdown_sources(docs)) == (2, 2, [])


@pytest.mark.parametrize(
    "entry",
    [
        "/outside.md",
        "../outside.md",
        "cases/../../outside.md",
        "C:/outside.md",
        "C:\\outside.md",
        "cases\\source.md",
        "index.html",
        "",
        " index.md",
        "index.md\n",
        "index\x00.md",
        None,
        5,
    ],
)
def test_invalid_manifest_paths_fail_before_audit(api, tmp_path, entry):
    """Paths are safe repository-relative Markdown on Unix and Windows alike."""
    docs, _ = tree(tmp_path)
    (docs / "publication-core.json").write_text(
        json.dumps({"schema_version": 1, "excluded_pages": [entry]})
    )
    with pytest.raises(ValueError, match="Invalid Markdown source"):
        api.excluded_markdown_sources(docs)


@pytest.mark.parametrize(
    "profile,message",
    [
        ([], "schema_version"),
        ({"schema_version": 2, "excluded_pages": []}, "schema_version"),
        ({"schema_version": True, "excluded_pages": []}, "schema_version"),
        ({"schema_version": 1.0, "excluded_pages": []}, "schema_version"),
        ({"schema_version": 1, "excluded_pages": "supplementary.md"}, "must be a list"),
        ({"schema_version": 1, "excluded_pages": ["unknown.md"]}, "missing or outside"),
        (
            {"schema_version": 1, "excluded_pages": ["supplementary.md", "supplementary.md"]},
            "Duplicate",
        ),
        (
            {"schema_version": 1, "canonical_pages": ["index.md"], "excluded_pages": ["index.md"]},
            "Canonical",
        ),
    ],
)
def test_manifest_structure_and_canonical_overlap_are_explicit(api, tmp_path, profile, message):
    """A broken profile is rejected rather than weakening generated-page requirements."""
    docs, _ = tree(tmp_path)
    (docs / "publication-core.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match=message):
        api.excluded_markdown_sources(docs)


def test_missing_and_malformed_profiles(api, tmp_path):
    """An absent profile means full checking; an unreadable profile is an error."""
    docs, _ = tree(tmp_path)
    manifest = docs / "publication-core.json"
    manifest.unlink()
    assert api.excluded_markdown_sources(docs) == set()
    manifest.write_text("{broken json")
    with pytest.raises(ValueError, match="Cannot read"):
        api.excluded_markdown_sources(docs)


def test_manifest_symlink_escape_is_rejected(api, tmp_path):
    """An existing source link outside docs cannot be declared a publication exclusion."""
    docs, _ = tree(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("Outside source.")
    try:
        (docs / "linked.md").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires platform support")
    (docs / "publication-core.json").write_text(
        json.dumps({"schema_version": 1, "excluded_pages": ["linked.md"]})
    )
    with pytest.raises(ValueError, match="outside docs"):
        api.excluded_markdown_sources(docs)
