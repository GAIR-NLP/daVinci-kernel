"""
SkillLibrary: on-disk versioned skill library (JSONL format).

File layout:
  <library_root>/
    global_step_0.jsonl      # initial snapshot (empty = no skills yet)
    global_step_50.jsonl     # snapshot after flush at step 50
    global_step_100.jsonl
    ...

Each .jsonl file: one JSON object per line, one skill per line:
  {"name": "vectorized_load_store", "description": "...",
   "scope": "general", "tags": [...], "content": "...", "verify_speedup": 1.5}

Skills are identified by their `name` field (unique within a snapshot).
In the public API, "rel_path" == skill name (kept for backward compatibility).

Checkpoint restart:
  load_for_step(N) loads the snapshot with max(step) <= N.
  The trainer passes meta_info["global_step"] on every generate_sequences() call,
  so resuming from checkpoint step M automatically picks up the correct snapshot.
"""

import json
import os
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkillMeta:
    """Metadata for one skill (used by selection agent prompt building)."""
    name: str
    description: str
    scope: str
    tags: List[str] = field(default_factory=list)
    rel_path: str = ""   # == name in JSONL format (kept for API compat)


@dataclass
class NewSkill:
    """A skill proposed by the Summary Agent, not yet persisted."""
    name: str
    description: str
    scope: str
    tags: List[str]
    content: str         # full markdown body
    verify_speedup: float = 0.0


# ---------------------------------------------------------------------------
# SkillLibrary
# ---------------------------------------------------------------------------

class SkillLibrary:
    """
    Versioned on-disk skill library backed by JSONL snapshots.

    Public API (unchanged from md-based version):
      load_for_step(global_step)
      list_rel_paths()          -> List[str]   (skill names)
      get_skill_meta(name)      -> Optional[SkillMeta]
      get_skill_content(name)   -> str          (formatted header + body)
      get_rel_path_by_name(name)-> Optional[str]
      retrieve_bm25(query, top_k)              -> List[str]
      get_file_tree_for_candidates(...)        -> str
      get_file_tree_for_selection(...)         -> str
      stage_new_skill(skill)
      flush_staged_skills(global_step)
    """

    def __init__(self, library_root: str, global_step_prefix: str = "global_step_"):
        self.library_root = library_root
        self.prefix = global_step_prefix
        self._snapshot_step: Optional[int] = None  # step of currently loaded snapshot
        self._initialized: bool = False             # True after first load_for_step attempt
        self._skills: List[dict] = []              # full skill records in order
        self._name_to_idx: Dict[str, int] = {}     # name -> index in _skills (O(1) lookup)
        self._meta_cache_dict: Optional[Dict[str, SkillMeta]] = None  # lazily built
        self._staged: List[NewSkill] = []
        self._bm25_index = None                    # BM25Okapi, built lazily
        self._bm25_names: List[str] = []           # parallel to BM25 corpus

    # ------------------------------------------------------------------
    # JSONL paths
    # ------------------------------------------------------------------

    def _jsonl_path(self, step: int) -> str:
        return os.path.join(self.library_root, f"{self.prefix}{step}.jsonl")

    def _list_checkpoints(self) -> List[int]:
        """Return sorted list of available snapshot step numbers."""
        if not os.path.isdir(self.library_root):
            return []
        steps = []
        for fname in os.listdir(self.library_root):
            if fname.startswith(self.prefix) and fname.endswith(".jsonl"):
                suffix = fname[len(self.prefix):-len(".jsonl")]
                if suffix.isdigit():
                    steps.append(int(suffix))
        return sorted(steps)

    # ------------------------------------------------------------------
    # Loading / versioning
    # ------------------------------------------------------------------

    def load_from_file(self, path: str) -> None:
        """
        Load skills directly from a specific JSONL file, bypassing step versioning.

        Used by eval_with_skill_grading where the caller specifies an exact snapshot
        file (e.g. skill_library/global_step_20.jsonl) rather than a directory.
        Sets _initialized=True so generate_sequences() won't call load_for_step().
        """
        self._reset_state()
        self._initialized = True
        if not os.path.exists(path):
            print(f"[SkillLibrary] load_from_file: file not found: {path}")
            return
        self._load_jsonl(path)
        self._snapshot_step = 0
        print(f"[SkillLibrary] load_from_file: loaded {len(self._skills)} skills from {path}")

    def load_for_step(self, global_step: int, start_step: int = 0) -> None:
        """
        Load the snapshot with the largest step number <= global_step,
        but never beyond start_step (to avoid loading snapshots written by a
        previous run that started from a later checkpoint than the current one).

        On checkpoint restart from step S:
          - Files with step <= S are valid initial state.
          - Files with step > S were written by a previous run and must be
            ignored; they will be overwritten as training progresses.
        """
        checkpoints = self._list_checkpoints()
        # ceiling: when start_step>0, never load snapshots beyond that point
        # (they were written by a previous run from a later checkpoint).
        # start_step=0 means fresh start — no upper bound restriction.
        ceiling = min(global_step, start_step) if start_step > 0 else global_step
        valid = [s for s in checkpoints if s <= ceiling]
        chosen_step = max(valid) if valid else None

        if chosen_step == self._snapshot_step:
            return  # already loaded

        self._reset_state()
        self._snapshot_step = chosen_step
        self._initialized = True  # mark as initialized even if library is empty

        if chosen_step is None:
            print(f"[SkillLibrary] load_for_step({global_step}, start={start_step}): "
                  f"no snapshot found — library empty (available: {checkpoints})")
            return

        path = self._jsonl_path(chosen_step)
        self._load_jsonl(path)
        print(f"[SkillLibrary] load_for_step({global_step}, start={start_step}): "
              f"loaded step={chosen_step}  skills={len(self._skills)}  path={path}")

    def _load_jsonl(self, path: str) -> None:
        """Read a JSONL file and populate _skills / _name_to_idx."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(d, dict) or not d.get("name"):
                    continue
                name = str(d["name"])
                if name in self._name_to_idx:
                    continue  # first occurrence wins on name collision
                self._name_to_idx[name] = len(self._skills)
                self._skills.append(d)

    def _reset_state(self) -> None:
        self._snapshot_step = None
        # Note: _initialized is NOT reset here — once initialized, always initialized.
        # It is only reset implicitly when a new SkillLibrary instance is created.
        self._skills = []
        self._name_to_idx = {}
        self._meta_cache_dict = None
        self._bm25_index = None
        self._bm25_names = []

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _bm25_tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[_\-/]", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [t for t in text.split() if t]

    def _ensure_bm25_index(self) -> None:
        if self._bm25_index is not None or not self._skills:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return

        corpus, names = [], []
        for s in self._skills:
            doc_text = " ".join([
                s.get("name", ""),
                s.get("description", ""),
                " ".join(s.get("tags", [])),
                s.get("content", ""),
            ])
            corpus.append(self._bm25_tokenize(doc_text))
            names.append(s["name"])

        self._bm25_index = BM25Okapi(corpus)
        self._bm25_names = names

    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[str]:
        """Return up to top_k skill names ordered by BM25 score."""
        if not self._skills:
            return []
        self._ensure_bm25_index()
        if self._bm25_index is None:
            # rank_bm25 not installed — return first top_k names
            return [s["name"] for s in self._skills[:top_k]]

        q_tokens = self._bm25_tokenize(query)
        scores = self._bm25_index.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._bm25_names[i] for i in ranked[:top_k] if scores[i] > 0]

    # ------------------------------------------------------------------
    # Querying — "rel_path" == name in JSONL world
    # ------------------------------------------------------------------

    def list_rel_paths(self) -> List[str]:
        """Return all skill names (= rel_paths in JSONL format)."""
        return [s["name"] for s in self._skills]

    def get_skill_meta(self, name: str) -> Optional[SkillMeta]:
        idx = self._name_to_idx.get(name)
        if idx is None:
            return None
        s = self._skills[idx]
        meta = SkillMeta(
            name=s.get("name", ""),
            description=s.get("description", ""),
            scope=s.get("scope", "general"),
            tags=s.get("tags", []) if isinstance(s.get("tags"), list) else [],
        )
        meta.rel_path = s["name"]   # rel_path == name
        return meta

    @property
    def _meta_cache(self) -> Dict[str, SkillMeta]:
        """
        Lazy dict of name -> SkillMeta.
        Used by skill_prompt_builder._format_file_tree_for_selection.
        Rebuilt when skills change.
        """
        if self._meta_cache_dict is None:
            self._meta_cache_dict = {
                s["name"]: self.get_skill_meta(s["name"])   # type: ignore[arg-type]
                for s in self._skills
            }
        return self._meta_cache_dict   # type: ignore[return-value]

    def get_skill_content(self, name: str) -> str:
        """
        Return formatted text for skill `name`:
          frontmatter-style header + markdown body.
        Used for injecting skill text into policy prompts.
        Raises FileNotFoundError if not found.
        """
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise FileNotFoundError(f"Skill not found: {name!r}")
        s = self._skills[idx]
        tags_str = "[" + ", ".join(s.get("tags", [])) + "]" if s.get("tags") else "[]"
        return (
            f"---\n"
            f"name: {s['name']}\n"
            f"description: {s.get('description', '')}\n"
            f"scope: {s.get('scope', 'general')}\n"
            f"tags: {tags_str}\n"
            f"---\n\n"
            f"{s.get('content', '').strip()}\n"
        )

    def get_rel_path_by_name(self, name: str) -> Optional[str]:
        """In JSONL format, rel_path == name. Returns name if it exists."""
        return name if name in self._name_to_idx else None

    def get_file_tree_for_candidates(
        self,
        candidate_rel_paths: Optional[List[str]] = None,
        max_skills: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Compat wrapper — delegates to get_file_tree_for_selection."""
        return self.get_file_tree_for_selection(
            candidate_names=candidate_rel_paths,
            max_skills=max_skills,
            seed=seed,
        )

    def get_file_tree_for_selection(
        self,
        candidate_names: Optional[List[str]] = None,
        max_skills: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Return a numbered skill list for the selection agent.
        Format matches selected_skill_data.py _format_candidates():
          N. **name**  [scope=...]  [tags: ...]
             description
        """
        if not self._skills:
            return "(skill library is empty)"

        if candidate_names:
            items = [
                self._skills[self._name_to_idx[n]]
                for n in candidate_names
                if n in self._name_to_idx
            ]
        else:
            items = list(self._skills)

        if max_skills is not None and max_skills < len(items):
            import random as _random
            rng = _random.Random(seed)
            items = rng.sample(items, max_skills)

        lines = []
        for i, s in enumerate(items, 1):
            tags = ", ".join(s.get("tags", []))
            scope = s.get("scope", "general")
            lines.append(
                f"{i:2d}. **{s['name']}**  [scope={scope}]"
                + (f"  [tags: {tags}]" if tags else "")
            )
            lines.append(f"    {s.get('description', '')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Staging / flushing
    # ------------------------------------------------------------------

    def stage_new_skill(self, skill: NewSkill) -> None:
        """Buffer a new skill until flush_staged_skills is called."""
        self._staged.append(skill)

    def set_skills(self, skills: list, snapshot_step: int) -> None:
        """Overwrite in-memory skill list directly (used for broadcast after flush)."""
        self._reset_state()
        self._snapshot_step = snapshot_step
        self._initialized = True
        for s in skills:
            name = s["name"]
            self._name_to_idx[name] = len(self._skills)
            self._skills.append(s)

    def flush_staged_skills(self, global_step: int) -> None:
        """
        Write all staged skills to a new JSONL snapshot.

        Always writes a new snapshot (even if _staged is empty) so that the
        versioned timeline is complete and load_for_step() can always find a
        snapshot for the current step.

        Algorithm:
          1. Start with all skills from the current snapshot.
          2. For each staged skill:
             - Deduplicate name: if it already exists, append _1, _2 …
             - Append to the list.
          3. Atomic write: write to .tmp then os.replace → new JSONL file.
          4. Reload in-memory state.
        """
        new_path = self._jsonl_path(global_step)
        os.makedirs(self.library_root, exist_ok=True)

        # Build new skill list: existing + staged (with name dedup)
        new_skills: List[dict] = list(self._skills)
        existing_names: set = {s["name"] for s in new_skills}
        n_added = 0

        for skill in self._staged:
            base = re.sub(r"[^\w\-]", "_", skill.name.lower())
            name = base
            suffix = 1
            while name in existing_names:
                name = f"{base}_{suffix}"
                suffix += 1

            new_skills.append({
                "name":           name,
                "description":    skill.description,
                "scope":          skill.scope,
                "tags":           skill.tags,
                "content":        skill.content,
                "verify_speedup": skill.verify_speedup,
            })
            existing_names.add(name)
            n_added += 1

        # Atomic write
        tmp_path = new_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for s in new_skills:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        os.replace(tmp_path, new_path)

        # Update in-memory state directly (no disk re-read)
        self._reset_state()
        self._snapshot_step = global_step
        for s in new_skills:
            name = s["name"]
            self._name_to_idx[name] = len(self._skills)
            self._skills.append(s)
        self._staged = []

        print(f"[SkillLibrary] flush_staged_skills: "
              f"step={global_step}  added={n_added}  total={len(self._skills)}  "
              f"path={new_path}")
        return {
            "skill/library_size":   len(self._skills),
            "skill/new_this_step":  n_added,
            "skill/flush_skipped":  False,
        }
