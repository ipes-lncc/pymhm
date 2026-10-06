"""Verify release preparation against real private Git histories and metadata files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import release
from scripts.check_metadata import VERSION_FIELDS, validate_metadata

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> str:
    """Run real Git in a private repository without inheriting signing settings."""
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _commit(root: Path, message: str) -> str:
    """Create a private commit with its explicit staging and local identity."""
    _git(root, "add", ".")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Provide synchronized metadata and two commits on a fetched main reference."""
    files = {
        "pyproject.toml": (
            '[project]\nname = "pymhm"\nversion = "0.1.0"\ndescription = "test"\n'
            'requires-python = ">=3.11"\ndependencies = ["numpy>=1.26"]\n'
        ),
        "pixi.toml": (
            '[workspace]\nname = "pymhm"\nversion = "0.1.0"\ndescription = "test"\n'
            '[dependencies]\npython = ">=3.11"\nnumpy = ">=1.26"\n'
        ),
        "src/pymhm/__init__.py": '__version__ = "0.1.0"\n',
        "CITATION.cff": 'version: "0.1.0"\n',
        "recipe/recipe.yaml": 'context:\n  version: "0.1.0"\n',
        "README.md": (
            "[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-21918c.svg)](url)\n"
            "Version 0.1.0 is an official release of PyMHM.\n"
        ),
        "docs/index.md": "This is version 0.1.0, an official release of PyMHM.\n",
        "CHANGELOG.md": "# Changelog\n\n## 0.1.0\n\n- Original release notes.\n",
        "cliff.toml": (ROOT / "cliff.toml").read_text(encoding="utf-8"),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    _git(tmp_path, "init")
    _git(tmp_path, "checkout", "-b", "main")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "core.autocrlf", "false")
    _commit(tmp_path, "feat: initial implementation")
    _git(tmp_path, "tag", "v0.1.0")
    (tmp_path / "feature.txt").write_text("new feature\n", encoding="utf-8")
    _commit(tmp_path, "fix: preserve reconstructed fields")
    _git(tmp_path, "update-ref", release.MAIN_REF, "HEAD")
    return tmp_path


@pytest.fixture
def cliff(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the generator contract while all repository commands use real Git."""
    calls: list[list[str]] = []
    original = subprocess.run

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Supply deterministic notes for controlled preflight and rollback failures."""
        if command[0] == "git-cliff":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "### Fixes\n\n- Preserve fields\n", "")
        return original(command, **kwargs)

    monkeypatch.setattr(release.subprocess, "run", run)
    return calls


def _snapshot(root: Path) -> dict[str, bytes]:
    """Capture every release-owned file for preflight and rollback comparisons."""
    return {
        relative: (root / relative).read_bytes() for relative in (*VERSION_FIELDS, "CHANGELOG.md")
    }


def test_release_version_policy_and_prerelease_ordering() -> None:
    versions = ["1.2.3a9", "1.2.3a10", "1.2.3b0", "1.2.3rc1", "1.2.3", "1.2.4"]
    assert sorted(reversed(versions), key=release.version_key) == versions
    assert release.is_prerelease("1.2.3rc1")
    assert not release.is_prerelease("1.2.3")
    for invalid in ("v1.2.3", "01.2.3", "1.2", "1.2.3-rc.1", "1.2.3rc01", "1.2.3+cpu"):
        with pytest.raises(ValueError, match="Invalid version"):
            release.version_key(invalid)


def test_native_cliff_uses_main_range_and_preserves_all_commit_categories(repository: Path) -> None:
    (repository / "breaking.txt").write_text("new operator\n", encoding="utf-8")
    _commit(repository, "feat!: expose universal operators\n\nBREAKING CHANGE: use explicit forms")
    (repository / "unconventional.txt").write_text("trace orientation\n", encoding="utf-8")
    _commit(repository, "Generalize trace orientation")
    main = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", release.MAIN_REF, "HEAD")
    base = _git(repository, "rev-parse", "v0.1.0")
    _git(repository, "checkout", "-b", "unrelated", base)
    (repository / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    _commit(repository, "feat: unrelated branch")
    _git(repository, "tag", "-a", "v999.0.0", "-m", "Unmerged release")
    _git(repository, "checkout", "main")
    _git(repository, "tag", "not-a-release", base)
    _git(repository, "tag", "v0.1.1rc1", base)
    _git(repository, "tag", "v01.2.3", base)
    _git(repository, "checkout", "-b", "release-preparation")
    (repository / "preparation.txt").write_text("preparation\n", encoding="utf-8")
    _commit(repository, "chore: prepare release")
    source, notes = release.generate_changelog(repository)
    assert source == release.ChangelogSource("v0.1.1rc1", base, main)
    assert f"<!-- Source: v0.1.1rc1..{main} -->" in notes
    assert "### Features" in notes and "### Fixes" in notes and "### Other changes" in notes
    assert "**Breaking:** expose universal operators" in notes
    assert "Generalize trace orientation" in notes
    assert "unrelated branch" not in notes and "prepare release" not in notes
    assert release.prepare_release(repository, "0.2.0") == source
    assert release.check_release(repository)[0] == "0.2.0"


def test_prepare_syncs_every_version_and_preserves_previous_and_manual_notes(
    repository: Path, cliff: list[list[str]]
) -> None:
    previous = (repository / "CHANGELOG.md").read_bytes()
    path = repository / "CHANGELOG.md"
    path.write_bytes(
        previous.replace(
            b"# Changelog\n\n",
            b"# Changelog\n\n## Unreleased\n\n- User-written pending feature.\n\n",
            1,
        )
    )
    assert release.check_release(repository) == ("0.1.0", "- Original release notes.")
    readme = repository / "README.md"
    readme.write_bytes(readme.read_bytes().replace(b"\n", b"\r\n"))
    release.prepare_release(repository, "1.0.0")
    assert validate_metadata(repository) == "1.0.0"
    assert b"Version 1.0.0 is an official release of PyMHM.\r\n" in readme.read_bytes()
    assert b"version-1.0.0-21918c.svg" in readme.read_bytes()
    assert b"Version: 1.0.0" in readme.read_bytes()
    assert b"0.1.0" not in readme.read_bytes()
    changelog_path = repository / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    assert "## Unreleased" not in changelog
    assert changelog.count("- User-written pending feature.") == 1
    assert changelog.endswith(previous.decode().split("\n\n", 1)[1])
    changelog_path.write_text(
        changelog.replace(
            release.GENERATED_END, release.GENERATED_END + "\n\n- Manual migration note."
        ),
        encoding="utf-8",
    )
    release.prepare_release(repository, "1.0.0")
    assert "- Manual migration note." in changelog_path.read_text(encoding="utf-8")
    assert changelog_path.read_text(encoding="utf-8").count("- User-written pending feature.") == 1
    assert cliff[0][-1] == cliff[1][-1]
    assert _git(repository, "tag", "--list") == "v0.1.0"
    assert release.check_release(repository)[0] == "1.0.0"
    pending = _snapshot(repository)
    with pytest.raises(ValueError, match="untagged release"):
        release.prepare_release(repository, "1.0.1")
    assert _snapshot(repository) == pending


def test_initial_release_is_explicit_and_records_real_main_history(
    repository: Path, cliff: list[list[str]]
) -> None:
    with pytest.raises(ValueError, match="--initial is allowed only"):
        release.changelog_source(repository, initial=True)
    _git(repository, "tag", "-d", "v0.1.0")
    original = _snapshot(repository)
    with pytest.raises(ValueError, match="use --initial"):
        release.prepare_release(repository, "0.2.0")
    assert _snapshot(repository) == original
    path = repository / "CHANGELOG.md"
    path.write_text(
        f"# Changelog\n\n## 0.1.0\n\n{release.GENERATED_START}\n"
        f"{release.GENERATED_END}\n\n- Handwritten first-release notes.\n",
        encoding="utf-8",
    )
    source = release.prepare_release(repository, "0.1.0", initial=True)
    assert source.tag is None and source.base_commit is None
    assert cliff[-1][-1] == source.main_commit
    assert f"origin/main@{source.main_commit} (initial release)" in path.read_text()
    assert "Handwritten first-release notes." in path.read_text()
    assert _git(repository, "tag", "--list") == ""


def test_prepare_preflight_rejects_existing_tags_old_versions_and_metadata_drift(
    repository: Path, cliff: list[list[str]]
) -> None:
    original = _snapshot(repository)
    for version, message in (("0.1.0", "already exists"), ("0.0.9", "must be greater")):
        with pytest.raises(ValueError, match=message):
            release.prepare_release(repository, version)
        assert _snapshot(repository) == original
    citation = repository / "CITATION.cff"
    citation.write_text('version: "9.9.9"\n', encoding="utf-8")
    inconsistent = _snapshot(repository)
    with pytest.raises(ValueError, match="CITATION.cff: version"):
        release.prepare_release(repository, "0.2.0")
    assert _snapshot(repository) == inconsistent
    citation.write_bytes(original["CITATION.cff"])
    readme = repository / "README.md"
    official = b"Version 0.1.0 is an official release of PyMHM.\n"
    for invalid, message in (
        (original["README.md"].replace(official, official.replace(b"0.1.0", b"9.9.9")), "version"),
        (original["README.md"].replace(official, b""), "expected one"),
        (original["README.md"] + official, "expected one"),
    ):
        readme.write_bytes(invalid)
        inconsistent = _snapshot(repository)
        with pytest.raises(ValueError, match=f"README.md: {message}"):
            release.prepare_release(repository, "1.0.0")
        assert _snapshot(repository) == inconsistent
    readme.write_bytes(original["README.md"])
    main = _git(repository, "rev-parse", release.MAIN_REF)
    _git(repository, "checkout", "--detach", "v0.1.0")
    with pytest.raises(ValueError, match="branch must include origin/main"):
        release.prepare_release(repository, "0.2.0")
    assert _snapshot(repository) == original
    _git(repository, "checkout", "main")
    shallow = repository / ".git/shallow"
    shallow.write_text(main + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Full Git history"):
        release.changelog_source(repository)
    shallow.unlink()
    _git(repository, "update-ref", release.MAIN_REF, "v0.1.0")
    with pytest.raises(ValueError, match="No commits"):
        release.prepare_release(repository, "0.2.0")
    assert _snapshot(repository) == original
    _git(repository, "update-ref", "-d", release.MAIN_REF)
    with pytest.raises(ValueError, match="origin/main is unavailable"):
        release.changelog_source(repository)


def test_preparation_rolls_back_all_files_when_replacement_fails(
    repository: Path, cliff: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _snapshot(repository)
    replace = os.replace
    failed = False

    def fail_once(source: Path, destination: Path) -> None:
        """Fail after the first update, then permit real rollback operations."""
        nonlocal failed
        if destination.name == "pixi.toml" and not failed:
            failed = True
            raise OSError("controlled replacement failure")
        replace(source, destination)

    monkeypatch.setattr(release.os, "replace", fail_once)
    with pytest.raises(OSError, match="controlled replacement failure"):
        release.prepare_release(repository, "0.2.0")
    assert _snapshot(repository) == original
    assert not list(repository.rglob(".*.????????"))


def test_checked_tag_extracts_notes_and_prerelease_cli_outputs(
    repository: Path, cliff: list[list[str]]
) -> None:
    release.prepare_release(repository, "0.2.0rc1")
    _commit(repository, "chore: prepare release 0.2.0rc1")
    _git(repository, "tag", "-a", "v0.2.0rc1", "-m", "Release 0.2.0rc1")
    assert release.check_release(repository, "v0.2.0rc1")[0] == "0.2.0rc1"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.release",
            "check",
            "--tag",
            "v0.2.0rc1",
            "--notes-output",
            "build/notes.md",
            "--github-output",
            "github-output",
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Validated release 0.2.0rc1" in result.stdout
    assert "- Preserve fields" in (repository / "build/notes.md").read_text()
    assert (repository / "github-output").read_text() == "version=0.2.0rc1\nprerelease=true\n"
    (repository / "later.txt").write_text("later commit\n", encoding="utf-8")
    _commit(repository, "feat: after release")
    with pytest.raises(ValueError, match="checked-out commit"):
        release.check_release(repository, "v0.2.0rc1")
    with pytest.raises(ValueError, match="Expected tag"):
        release.check_release(repository, "v0.2.0")


def test_changelog_updates_reject_rewriting_old_or_unmarked_entries() -> None:
    block = f"{release.GENERATED_START}\n- Generated note.\n{release.GENERATED_END}"
    changelog = f"# Changelog\n\n## 0.2.0\n\n{block}\n\n## 0.1.0\n\n- Old release.\n"
    preamble = "# Changelog\n\nRelease and compatibility notes for PyMHM.\n\n"
    introduced = changelog.replace("# Changelog\n\n", preamble)
    updated = release.update_changelog(introduced, "0.3.0", block)
    assert updated.startswith(preamble + "## 0.3.0\n")
    assert updated.endswith(changelog.split("\n\n", 1)[1])
    assert "compatibility notes" not in release.release_notes(updated, "0.3.0")
    pending = (
        preamble + "## Unreleased\n\n- Pending manual addition.\n\n" + changelog.split("\n\n", 1)[1]
    )
    assert release.release_notes(pending, "0.2.0") == block
    migrated = release.update_changelog(pending, "0.3.0", block)
    assert migrated.startswith(preamble + "## 0.3.0\n")
    assert release.release_notes(migrated, "0.3.0") == block + "\n\n- Pending manual addition."
    assert migrated.endswith(changelog.split("\n\n", 1)[1])
    assert release.update_changelog(migrated, "0.3.0", block) == migrated
    # A manually edited git-cliff preview contributes only its handwritten notes.
    preview = pending.replace(
        "- Pending manual addition.", block + "\n\n- Pending manual addition."
    )
    moved = release.update_changelog(preview, "0.2.0", block)
    assert moved.count(release.GENERATED_START) == 1
    assert release.release_notes(moved, "0.2.0") == block + "\n\n- Pending manual addition."
    assert release.update_changelog(moved, "0.2.0", block) == moved
    windows = release.update_changelog(
        pending.replace("\n", "\r\n"), "0.3.0", block.replace("\n", "\r\n")
    )
    assert "\r\r\n" not in windows and "\n" not in windows.replace("\r\n", "")
    assert windows == migrated.replace("\n", "\r\n")
    first = release.update_changelog(
        "# Changelog\n\n## Unreleased\n\n- Initial manual note.\n", "0.1.0", block
    )
    assert release.release_notes(first, "0.1.0") == block + "\n\n- Initial manual note."
    with pytest.raises(ValueError, match="older changelog"):
        release.update_changelog(changelog, "0.1.0", block)
    for invalid in (
        changelog.replace(release.GENERATED_START, ""),
        changelog.replace(release.GENERATED_END, release.GENERATED_END * 2),
    ):
        with pytest.raises(ValueError, match="exactly one generated block"):
            release.update_changelog(invalid, "0.2.0", block)
    for notes in ("", "- TODO: describe the release."):
        with pytest.raises(ValueError, match="complete release notes"):
            release.release_notes(f"# Changelog\n\n## 0.2.0\n\n{notes}", "0.2.0")
    for invalid in (
        pending.replace("## 0.1.0", "## Unreleased"),
        changelog + "\n## Unreleased\n\n- Misplaced note.\n",
        pending.replace("- Pending manual addition.", release.GENERATED_START),
    ):
        with pytest.raises(ValueError):
            release.update_changelog(invalid, "0.3.0", block)
