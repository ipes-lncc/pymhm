"""Prepare releases from origin/main with git-cliff and synchronized metadata."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scripts.check_metadata import validate_metadata, version_files

MAIN_REF = "refs/remotes/origin/main"
GENERATED_START = "<!-- pymhm:generated:start -->"
GENERATED_END = "<!-- pymhm:generated:end -->"
VERSION_PATTERN = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:(a|b|rc)(0|[1-9][0-9]*))?"


@dataclass(frozen=True)
class ChangelogSource:
    """Identify the exact main-branch history used to generate release notes."""

    tag: str | None
    base_commit: str | None
    main_commit: str

    @property
    def commit_range(self) -> str:
        """Return an explicit Git range, or the main tip for an initial release."""
        if self.base_commit is None:
            return self.main_commit
        return f"{self.base_commit}..{self.main_commit}"


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Validate canonical Python release notation and return its ordering key.

    Release versions use ``X.Y.Z``, optionally followed by ``aN``, ``bN`` or
    ``rcN``. Git tags add only a leading ``v``; Python, Conda and citation
    metadata retain the same version string. Development, post-release, local
    build and noncanonical spellings are rejected rather than normalized.
    """
    match = re.fullmatch(VERSION_PATTERN, version)
    if match is None:
        raise ValueError(f"Invalid version {version!r}; use X.Y.Z, X.Y.ZaN, X.Y.ZbN or X.Y.ZrcN")
    stage = {"a": 0, "b": 1, "rc": 2, None: 3}[match[4]]
    return int(match[1]), int(match[2]), int(match[3]), stage, int(match[5] or 0)


def is_prerelease(version: str) -> bool:
    """Return whether a canonical release version denotes an alpha, beta or RC."""
    return version_key(version)[3] < 3


def _git(root: Path, *arguments: str) -> str:
    """Execute Git without a shell and return stripped stdout or a useful error."""
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"Git command failed: {' '.join(arguments)}")
    return result.stdout.rstrip("\r\n")


def changelog_source(root: Path, *, initial: bool = False) -> ChangelogSource:
    """Select the nearest canonical release on origin/main's first-parent history.

    Tags on unrelated branches and preparation-branch commits cannot alter the
    selected range. An explicit ``initial`` request is required when that
    history contains no release tag; it uses the complete main-branch history.
    """
    if _git(root, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("Full Git history is required; fetch --unshallow, then run release-fetch")
    try:
        main_commit = _git(root, "rev-parse", "--verify", f"{MAIN_REF}^{{commit}}")
    except ValueError as error:
        raise ValueError("origin/main is unavailable; run the release-fetch task first") from error
    tags: dict[str, str] = {}
    refs = _git(
        root,
        "for-each-ref",
        "--format=%(refname:strip=2)\t%(objectname)\t%(*objectname)",
        "refs/tags",
    )
    for line in refs.splitlines():
        tag, object_name, peeled = line.split("\t")
        if not tag.startswith("v"):
            continue
        try:
            key = version_key(tag[1:])
        except ValueError:
            continue
        commit = peeled or object_name
        existing = tags.get(commit)
        if existing is None or key > version_key(existing[1:]):
            tags[commit] = tag
    for commit in _git(root, "rev-list", "--first-parent", main_commit).splitlines():
        if commit in tags:
            if initial:
                raise ValueError("--initial is allowed only before the first main-branch release")
            return ChangelogSource(tags[commit], commit, main_commit)
    if not initial:
        raise ValueError(
            "No release tag exists on origin/main; use --initial for the first release"
        )
    return ChangelogSource(None, None, main_commit)


def generate_changelog(root: Path, *, initial: bool = False) -> tuple[ChangelogSource, str]:
    """Generate one marked block with offline git-cliff and immutable range provenance."""
    source = changelog_source(root, initial=initial)
    if source.base_commit == source.main_commit:
        raise ValueError(
            "No commits on origin/main after its last release; preparation is excluded"
        )
    try:
        result = subprocess.run(
            [
                "git-cliff",
                "--config",
                str(root / "cliff.toml"),
                "--offline",
                "--no-exec",
                "--strip",
                "all",
                source.commit_range,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ValueError("git-cliff is missing; use the locked Pixi release environment") from error
    if result.returncode:
        raise ValueError(f"git-cliff failed: {result.stderr.strip()}")
    body = result.stdout.strip()
    if not body:
        raise ValueError("git-cliff produced no release notes; inspect cliff.toml")
    provenance = (
        f"{source.tag}..{source.main_commit}"
        if source.tag is not None
        else f"origin/main@{source.main_commit} (initial release)"
    )
    notes = f"{GENERATED_START}\n<!-- Source: {provenance} -->\n\n{body}\n\n{GENERATED_END}"
    return source, notes


def _releases(changelog: str) -> list[re.Match[str]]:
    """Locate release headings, allowing one separate leading Unreleased section."""
    if not re.match(r"\A# Changelog(?:\r?\n|\Z)", changelog):
        raise ValueError("CHANGELOG.md must start with # Changelog")
    headings = list(re.finditer(r"(?m)^## (?P<version>\S+)\r?$", changelog))
    pending = [
        index for index, heading in enumerate(headings) if heading["version"] == "Unreleased"
    ]
    if pending and pending != [0]:
        raise ValueError("CHANGELOG.md permits only one leading ## Unreleased section")
    return headings[1:] if pending else headings


def _generated_span(body: str) -> tuple[int, int]:
    """Select one complete generated block without discarding handwritten notes."""
    first, last = body.find(GENERATED_START), body.find(GENERATED_END)
    if (
        first < 0
        or last < first
        or body.count(GENERATED_START) != 1
        or body.count(GENERATED_END) != 1
    ):
        raise ValueError(
            "The current release needs exactly one generated block; preserve its markers"
        )
    return first, last + len(GENERATED_END)


def release_notes(changelog: str, version: str) -> str:
    """Return the first actual release's notes, excluding pending Unreleased changes."""
    releases = _releases(changelog)
    if not releases or releases[0]["version"] != version:
        raise ValueError(f"The first CHANGELOG.md release must be ## {version}")
    end = releases[1].start() if len(releases) > 1 else len(changelog)
    notes = changelog[releases[0].end() : end].strip()
    content = re.sub(r"<!--.*?-->", "", notes, flags=re.DOTALL).strip()
    if not content or re.search(r"(?m)^\s*-\s+TODO(?:\s|:|$)", content):
        raise ValueError("Write complete release notes before releasing")
    return notes


def update_changelog(changelog: str, version: str, notes: str) -> str:
    """Consume pending notes and refresh one release without changing historical records."""
    releases = _releases(changelog)
    newline = "\r\n" if "\r\n" in changelog else "\n"
    notes = notes.replace("\r\n", "\n").replace("\n", newline)
    pending = ""
    unreleased = re.search(r"(?m)^## Unreleased\r?$", changelog)
    if unreleased is not None:
        end = releases[0].start() if releases else len(changelog)
        pending = changelog[unreleased.end() : end].strip()
        if GENERATED_START in pending or GENERATED_END in pending:
            first, last = _generated_span(pending)
            pending = (pending[:first] + pending[last:]).strip()
        changelog = changelog[: unreleased.start()] + changelog[end:]
        releases = _releases(changelog)
    current = next((release for release in releases if release["version"] == version), None)
    if current is None:
        position = releases[0].start() if releases else len(changelog)
        preamble = changelog[:position]
        separator = "" if preamble.endswith(newline * 2) else newline
        if not preamble.endswith(newline):
            separator += newline
        body = notes + (newline * 2 + pending if pending else "")
        return (
            f"{preamble}{separator}## {version}{newline}{newline}"
            f"{body}{newline}{newline}{changelog[position:]}"
        )
    if current is not releases[0]:
        raise ValueError("Do not rewrite an older changelog release")
    end = releases[1].start() if len(releases) > 1 else len(changelog)
    body = changelog[current.start() : end]
    first, last = _generated_span(body)
    updated = body[:first] + notes + body[last:]
    if pending:
        updated = updated.rstrip() + newline * 2 + pending + newline * 2
    return changelog[: current.start()] + updated + changelog[end:]


def _stage_file(path: Path, data: bytes, mode: int) -> Path:
    """Stage closed replacement bytes next to the destination for portable atomic renaming."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
            os.chmod(temporary, mode)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    assert temporary is not None
    return temporary


def _write_updates(updates: dict[Path, bytes]) -> None:
    """Preflight all replacements, preserve permissions and roll back on write failure."""
    original = {path: path.read_bytes() for path in updates}
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in updates}
    staged: dict[Path, Path] = {}
    written: list[Path] = []
    try:
        for path, data in updates.items():
            staged[path] = _stage_file(path, data, modes[path])
        if any(path.read_bytes() != data for path, data in original.items()):
            raise ValueError(
                "Release files changed during preparation; retry from the current files"
            )
        try:
            for path, temporary in staged.items():
                os.replace(temporary, path)
                written.append(path)
        except BaseException:
            for path in reversed(written):
                temporary = _stage_file(path, original[path], modes[path])
                try:
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def prepare_release(root: Path, version: str, *, initial: bool = False) -> ChangelogSource:
    """Prepare all current-version fields and changelog after complete release preflight.

    This operation creates no commit, tag, GitHub release or publication. The
    executing branch must include origin/main; pending releases may be regenerated
    at the same version while tagged releases and older sections stay immutable.
    """
    version_key(version)
    if _git(root, "tag", "--list", f"v{version}"):
        raise ValueError(f"Tag v{version} already exists; never rewrite a tagged release")
    current = validate_metadata(root)
    source, notes = generate_changelog(root, initial=initial)
    if source.tag is not None and version_key(version) <= version_key(source.tag[1:]):
        raise ValueError(f"The new version must be greater than {source.tag[1:]}")
    try:
        _git(root, "merge-base", "--is-ancestor", source.main_commit, "HEAD")
    except ValueError as error:
        raise ValueError(
            "Your branch must include origin/main; merge or rebase before preparing"
        ) from error
    path = root / "CHANGELOG.md"
    changelog = path.read_bytes().decode("utf-8")
    if (
        current != version
        and not _git(root, "tag", "--list", f"v{current}")
        and "<!-- Source:" in changelog
    ):
        raise ValueError(f"An untagged release ({current}) is already being prepared")
    updated = update_changelog(changelog, version, notes)
    release_notes(updated, version)
    updates = version_files(root, current, version)
    updates[path] = updated.encode("utf-8")
    _write_updates(updates)
    return source


def check_release(root: Path, tag: str | None = None) -> tuple[str, str]:
    """Validate synchronized versions and usable release notes, optionally at an actual tag."""
    version = validate_metadata(root, tag)
    version_key(version)
    notes = release_notes((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    if tag is not None:
        tagged = _git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
        if tagged != _git(root, "rev-parse", "HEAD"):
            raise ValueError("The requested release tag must point to the checked-out commit")
    return version, notes


def main() -> None:
    """Dispatch release preparation, changelog preview and publication validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="Preview notes from origin/main without edits")
    preview.add_argument(
        "--initial", action="store_true", help="Use main's entire first-release history"
    )
    prepare = commands.add_parser("prepare", help="Synchronize versions and generate CHANGELOG.md")
    prepare.add_argument("version")
    prepare.add_argument(
        "--initial", action="store_true", help="Prepare the first main-branch release"
    )
    check = commands.add_parser("check", help="Validate metadata and release notes")
    check.add_argument("--tag")
    check.add_argument("--notes-output", type=Path)
    check.add_argument(
        "--github-output", type=Path, help="Append version/prerelease GitHub job outputs"
    )
    args = parser.parse_args()
    root = Path.cwd()
    try:
        if args.command == "preview":
            _, notes = generate_changelog(root, initial=args.initial)
            print(f"## Unreleased\n\n{notes}")
        elif args.command == "prepare":
            source = prepare_release(root, args.version, initial=args.initial)
            print(
                f"Prepared {args.version} from {source.commit_range}; "
                "review CHANGELOG.md and run version-check. No commit or tag was created."
            )
        else:
            version, notes = check_release(root, args.tag)
            if args.notes_output is not None:
                args.notes_output.parent.mkdir(parents=True, exist_ok=True)
                args.notes_output.write_text(notes + "\n", encoding="utf-8")
            if args.github_output is not None:
                with args.github_output.open("a", encoding="utf-8") as file:
                    file.write(
                        f"version={version}\nprerelease={str(is_prerelease(version)).lower()}\n"
                    )
            print(f"Validated release {version}: synchronized versions and CHANGELOG.md")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
