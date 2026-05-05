"""Skill library: storage, retrieval, and management of reusable skills."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from skilleval.core.types import SkillOrigin, SkillSpec
from skilleval.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _render_default_skill_md(spec: "SkillSpec") -> str:
    """Fallback SKILL.md body for skills created without a markdown source."""
    return (
        f"---\nname: {spec.name}\n"
        f"description: {spec.description}\n---\n\n"
        f"# {spec.name}\n\n{spec.description}\n"
    )


class SkillLibrary:
    """In-memory skill library with persistence, retrieval, and pruning.

    This is the runtime representation of the skill set S in the paper.
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._usage_counts: dict[str, int] = {}

    # -- CRUD ----------------------------------------------------------------

    def add(self, skill: BaseSkill) -> None:
        self._skills[skill.skill_id] = skill
        self._usage_counts.setdefault(skill.skill_id, 0)

    def remove(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)
        self._usage_counts.pop(skill_id, None)
        self._metadata.pop(skill_id, None)

    def get(self, skill_id: str) -> BaseSkill | None:
        return self._skills.get(skill_id)

    def list_skills(
        self, origin: SkillOrigin | None = None, level: int | None = None
    ) -> list[BaseSkill]:
        result = list(self._skills.values())
        if origin is not None:
            result = [s for s in result if s.origin == origin]
        if level is not None:
            result = [s for s in result if s.level == level]
        return result

    @property
    def size(self) -> int:
        return len(self._skills)

    # -- retrieval -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        required_tools: list[str] | None = None,
    ) -> list[BaseSkill]:
        """Retrieve the most relevant skills for a query.

        Default: keyword overlap on name + description + tool_calls.
        Override with embedding-based retrieval for production use.
        """
        scored: list[tuple[float, BaseSkill]] = []
        query_tokens = set(query.lower().split())
        for skill in self._skills.values():
            text = f"{skill.spec.name} {skill.spec.description} {' '.join(skill.spec.tool_calls)}"
            skill_tokens = set(text.lower().split())
            overlap = len(query_tokens & skill_tokens)
            if required_tools:
                tool_overlap = len(set(required_tools) & set(skill.spec.tool_calls))
                overlap += tool_overlap * 2
            scored.append((overlap, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def bundle(self, skill_ids: list[str]) -> list[BaseSkill]:
        """Return an ordered bundle of skills by ID (for OB policy)."""
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]

    # -- usage tracking & pruning -------------------------------------------

    def record_usage(self, skill_id: str) -> None:
        self._usage_counts[skill_id] = self._usage_counts.get(skill_id, 0) + 1

    def prune(self, min_usage: int = 1) -> list[str]:
        """Remove skills below the usage threshold; return pruned IDs."""
        if min_usage > 0 and sum(self._usage_counts.values()) == 0:
            logger.info(
                "Skipping prune because no usage has been recorded yet."
            )
            return []
        to_prune = [
            sid for sid, count in self._usage_counts.items()
            if count < min_usage
        ]
        for sid in to_prune:
            self.remove(sid)
        logger.info("Pruned %d skills below usage threshold %d", len(to_prune), min_usage)
        return to_prune

    def merge_duplicates(self, similarity_threshold: float = 0.9) -> int:
        """Merge near-duplicate skills; return count of merges."""
        skills = list(self._skills.values())
        merged = 0
        seen: set[str] = set()
        for i, s1 in enumerate(skills):
            if s1.skill_id in seen:
                continue
            for s2 in skills[i + 1:]:
                if s2.skill_id in seen:
                    continue
                if s1.similarity(s2) >= similarity_threshold:
                    keep = s1 if self._usage_counts.get(s1.skill_id, 0) >= self._usage_counts.get(s2.skill_id, 0) else s2
                    drop = s2 if keep is s1 else s1
                    self.remove(drop.skill_id)
                    seen.add(drop.skill_id)
                    merged += 1
        return merged

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save all skills to a single JSON file (legacy format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            sid: {
                "spec": skill.spec.to_dict(),
                "usage": self._usage_counts.get(sid, 0),
            }
            for sid, skill in self._skills.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_to_dir(self, dir_path: str | Path) -> None:
        """Save each skill as ``<skill_id>/SKILL.md`` plus a manifest.json.

        The SKILL.md body is the raw markdown produced by the writer
        (YAML frontmatter + markdown). The manifest records the origin
        and usage so the loader can reconstruct metadata without
        re-parsing every SKILL.md up front.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        entries = []
        for sid, skill in self._skills.items():
            skill_dir = dir_path / sid
            skill_dir.mkdir(parents=True, exist_ok=True)
            body = skill.spec.template or _render_default_skill_md(skill.spec)
            (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
            entries.append({
                "skill_id": sid,
                "path": f"{sid}/SKILL.md",
                "name": skill.spec.name,
                "description": skill.spec.description,
                "origin": skill.spec.origin.value,
                "level": skill.spec.level,
                "usage": self._usage_counts.get(sid, 0),
            })

        manifest = {"num_skills": len(entries), "skills": entries}
        with open(dir_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def save_to_markdown_dir(self, dir_path: str | Path) -> None:
        """Save each skill as SKILL.md in its own directory.

        Creates:
            dir_path/
              manifest.json     — skill list + metadata
              skills/
                <skill_name>/
                  SKILL.md      — YAML frontmatter + markdown body
        """
        dir_path = Path(dir_path)
        skills_dir = dir_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        skill_names = []
        skill_metadata = {}
        for sid, skill in self._skills.items():
            # Create skill directory using the skill name
            skill_dir = skills_dir / skill.spec.name
            skill_dir.mkdir(exist_ok=True)

            # Create human-readable title from snake_case name
            title = skill.spec.name.replace("_", " ").title()

            # Write SKILL.md with simple frontmatter (just name + description)
            with open(skill_dir / "SKILL.md", "w") as f:
                f.write("---\n")
                f.write(f"name: {skill.spec.name}\n")
                # Handle multi-line descriptions
                desc = skill.spec.description.replace("\n", " ").strip()
                f.write(f"description: {desc}\n")
                f.write("---\n\n")
                f.write(f"# {title}\n\n")
                f.write(skill.spec.template or "")

            skill_names.append(skill.spec.name)
            # Store extra metadata in manifest instead of frontmatter
            skill_metadata[skill.spec.name] = {
                "origin": skill.spec.origin.value if hasattr(skill.spec.origin, "value") else str(skill.spec.origin),
                "level": skill.spec.level,
                "usage": self._usage_counts.get(sid, 0),
                "tool_calls": skill.spec.tool_calls or [],
            }

        # Write manifest with metadata
        manifest = {
            "num_skills": len(skill_names),
            "skills": skill_names,
            "format": "markdown",
            "metadata": skill_metadata,
        }
        with open(dir_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved %d skills as markdown to %s", len(skill_names), dir_path)

    @classmethod
    def load_from_dir(cls, dir_path: str | Path) -> "SkillLibrary":
        """Load a library from a directory of ``<skill_id>/SKILL.md`` files."""
        from skilleval.skills.creation.top_down import PromptSkill
        from skilleval.skills.creation.llm_skill_writer import _parse_frontmatter

        dir_path = Path(dir_path)
        library = cls()

        manifest_path = dir_path / "manifest.json"
        entries: list[dict[str, Any]] = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            entries = manifest.get("skills", [])
            # Legacy manifests stored a bare list of skill ids.
            if entries and isinstance(entries[0], str):
                entries = [{"skill_id": sid, "path": f"{sid}/SKILL.md"} for sid in entries]
        else:
            for p in sorted(dir_path.glob("*/SKILL.md")):
                entries.append({"skill_id": p.parent.name, "path": f"{p.parent.name}/SKILL.md"})

        for entry in entries:
            sid = entry["skill_id"]
            md_path = dir_path / entry.get("path", f"{sid}/SKILL.md")
            if not md_path.exists():
                logger.warning("Skill file not found: %s", md_path)
                continue
            body = md_path.read_text(encoding="utf-8")
            meta, _rest = _parse_frontmatter(body)
            spec = SkillSpec(
                skill_id=sid,
                name=meta.get("name") or entry.get("name", sid),
                description=meta.get("description") or entry.get("description", ""),
                origin=SkillOrigin(entry.get("origin", "SD")),
                level=int(entry.get("level", 2)),
                tool_calls=[],
                template=body,
            )
            library.add(PromptSkill(spec))

        return library

    @classmethod
    def load_from_markdown_dir(cls, dir_path: str | Path) -> "SkillLibrary":
        """Load a library from a directory of SKILL.md files.

        Expects either:
            dir_path/
              manifest.json     — skill list + metadata (optional)
              skills/
                <skill_name>/
                  SKILL.md      — YAML frontmatter (name, description) + markdown body
        or:
            dir_path/
              skills/
                <skill_name>.md — YAML frontmatter (name, description) + markdown body
        """
        from skilleval.skills.creation.top_down import PromptSkill

        dir_path = Path(dir_path)
        library = cls()

        # Look for skills directory
        skills_dir = dir_path / "skills"
        if not skills_dir.exists():
            skills_dir = dir_path

        # Check manifest for skill list and metadata
        manifest_path = dir_path / "manifest.json"
        manifest_metadata = {}
        skill_files: list[tuple[str, Path]] = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            skill_names = manifest.get("skills", [])
            manifest_metadata = manifest.get("metadata", {})
            for skill_name in skill_names:
                nested_file = skills_dir / skill_name / "SKILL.md"
                flat_file = skills_dir / f"{skill_name}.md"
                if nested_file.exists():
                    skill_files.append((skill_name, nested_file))
                elif flat_file.exists():
                    skill_files.append((skill_name, flat_file))
                else:
                    logger.warning(
                        "Markdown skill file not found for %s under %s",
                        skill_name,
                        skills_dir,
                    )
        else:
            skill_files.extend(
                (d.name, d / "SKILL.md")
                for d in sorted(skills_dir.iterdir())
                if d.is_dir() and (d / "SKILL.md").exists()
            )
            skill_files.extend(
                (p.stem, p)
                for p in sorted(skills_dir.glob("*.md"))
                if p.name != "SKILL.md"
            )

        for skill_name, skill_file in skill_files:
            content = skill_file.read_text()

            # Parse YAML frontmatter
            if not content.startswith("---"):
                logger.warning("No YAML frontmatter in %s", skill_file)
                continue

            parts = content.split("---", 2)
            if len(parts) < 3:
                logger.warning("Malformed SKILL.md: %s", skill_file)
                continue

            # Simple frontmatter parsing (just name: and description:)
            frontmatter = {}
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    frontmatter[key.strip()] = val.strip()

            body = parts[2].strip()
            # Remove the # Title line if present (it's redundant with name)
            if body.startswith("#"):
                first_newline = body.find("\n")
                if first_newline > 0:
                    body = body[first_newline:].strip()

            # Get metadata from manifest or use defaults
            meta = manifest_metadata.get(skill_name, {})
            origin_str = meta.get("origin", "TD")
            try:
                origin = SkillOrigin(origin_str)
            except ValueError:
                origin = SkillOrigin.TRACE_DERIVED

            prefix = "td" if origin == SkillOrigin.TRACE_DERIVED else "sd"
            name = frontmatter.get("name", skill_name)
            skill_id = f"{prefix}_{name}"

            spec = SkillSpec(
                skill_id=skill_id,
                name=name,
                description=frontmatter.get("description", ""),
                origin=origin,
                level=meta.get("level", 2),
                tool_calls=meta.get("tool_calls") or [],
                template=body,
            )
            library.add(PromptSkill(spec))
            library._usage_counts[skill_id] = meta.get("usage", 0)

        logger.info("Loaded %d skills from markdown in %s", library.size, dir_path)
        return library

    @classmethod
    def load_auto(cls, dir_path: str | Path) -> "SkillLibrary":
        """Auto-detect format and load library from directory.

        Checks manifest.json for format hint, or looks for skills/ subdirectory.
        """
        dir_path = Path(dir_path)

        # Check manifest for format
        manifest_path = dir_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("format") == "markdown":
                return cls.load_from_markdown_dir(dir_path)

        # Check for skills/ directory (markdown format)
        if (dir_path / "skills").exists():
            return cls.load_from_markdown_dir(dir_path)

        # Also accept being pointed directly at the inner skills/ directory.
        if dir_path.name == "skills" and (
            any(dir_path.glob("*.md"))
            or any(d.is_dir() and (d / "SKILL.md").exists() for d in dir_path.iterdir())
        ):
            return cls.load_from_markdown_dir(dir_path)

        # Check for a directory containing multiple child libraries, for
        # example generated/redditv2/skills and generated/redial/skills.
        child_dirs = [
            p for p in sorted(dir_path.iterdir())
            if p.is_dir() and ((p / "manifest.json").exists() or (p / "skills").exists())
        ]
        if child_dirs:
            merged = cls()
            for child_dir in child_dirs:
                child = cls.load_auto(child_dir)
                for skill in child.list_skills():
                    merged.add(skill)
                    merged._usage_counts[skill.skill_id] = child._usage_counts.get(
                        skill.skill_id, 0
                    )
            return merged

        # Default to JSON format
        return cls.load_from_dir(dir_path)

    def summary(self) -> dict[str, Any]:
        by_origin: dict[str, int] = {}
        by_level: dict[int, int] = {}
        for s in self._skills.values():
            by_origin[s.origin.value] = by_origin.get(s.origin.value, 0) + 1
            by_level[s.level] = by_level.get(s.level, 0) + 1
        return {
            "total_skills": self.size,
            "by_origin": by_origin,
            "by_level": by_level,
            "total_usage": sum(self._usage_counts.values()),
        }
