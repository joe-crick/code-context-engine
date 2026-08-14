"""Keep a worktree overlay synchronized with its shared base index."""

from __future__ import annotations

from pathlib import Path

from context_engine.git.diff import GitWorktreeDiff
from context_engine.git.repository import resolve_git_repository_context
from context_engine.git.storage import RepositoryStorageLayout, repository_storage_layout
from context_engine.indexer.manifest import Manifest
from context_engine.indexer.pipeline import IndexResult, run_indexing
from context_engine.storage.local_backend import LocalBackend


def indexed_worktree_diff(
    layout: RepositoryStorageLayout,
    *,
    head_sha: str | None = None,
) -> GitWorktreeDiff | None:
    """Compare the indexed base snapshot with the indexed worktree snapshot."""
    base_path = layout.base_dir / "manifest.json"
    overlay_path = layout.worktree_dir / "manifest.json"
    if not base_path.exists() or not overlay_path.exists():
        return None

    base_manifest = Manifest(base_path)
    base = base_manifest.entries
    overlay = Manifest(overlay_path).entries
    shared = base.keys() & overlay.keys()
    return GitWorktreeDiff(
        modified={path for path in shared if base[path] != overlay[path]},
        added=set(overlay) - set(base),
        deleted=set(base) - set(overlay),
        base_sha=base_manifest.last_git_sha,
        head_sha=head_sha,
    )


async def sync_worktree_overlay(
    config,
    project_dir: str | Path,
    *,
    force: bool = False,
    log_fn=None,
    progress_fn=None,
    embed_progress_fn=None,
    phase_fn=None,
) -> tuple[IndexResult, GitWorktreeDiff, RepositoryStorageLayout]:
    """Scan a worktree but persist only files that differ from the shared base."""
    project_dir = Path(project_dir).resolve()
    context = resolve_git_repository_context(project_dir)
    if context is None:
        raise ValueError(f"not a Git worktree: {project_dir}")

    layout = repository_storage_layout(config, project_dir, context, migrate_legacy=False)
    base_path = layout.base_dir / "manifest.json"
    if not base_path.exists():
        raise FileNotFoundError(
            f"shared base index is missing at {layout.base_dir}; run cce index-base"
        )

    base_manifest = Manifest(base_path)
    overlay_manifest = Manifest(layout.worktree_dir / "manifest.json")
    overlay = LocalBackend(base_path=str(layout.worktree_dir))
    stored_paths = set(overlay.file_chunk_counts())
    if stored_paths and overlay_manifest.embedding_dim != base_manifest.embedding_dim:
        await overlay.clear()
        stored_paths.clear()
    base_entries = base_manifest.entries
    overlay_entries = overlay_manifest.entries
    snapshot = dict(base_entries)
    snapshot.update(overlay_entries)

    # A refreshed base can make a previously base-only worktree file diverge.
    # Remove its inherited hash so the existing pipeline indexes that file.
    for path, content_hash in overlay_entries.items():
        if base_entries.get(path) != content_hash and path not in stored_paths:
            snapshot.pop(path, None)

    if force:
        for path, content_hash in list(snapshot.items()):
            if base_entries.get(path) != content_hash:
                snapshot.pop(path)

    layout.worktree_dir.mkdir(parents=True, exist_ok=True)
    overlay_manifest.replace_entries(snapshot)
    overlay_manifest.embedding_dim = base_manifest.embedding_dim
    overlay_manifest.save()

    result = await run_indexing(
        config,
        project_dir,
        full=False,
        storage_base_override=layout.worktree_dir,
        log_fn=log_fn,
        progress_fn=progress_fn,
        embed_progress_fn=embed_progress_fn,
        phase_fn=phase_fn,
    )
    diff = indexed_worktree_diff(layout, head_sha=context.head_sha) or GitWorktreeDiff()
    if result.errors:
        return result, diff, layout

    stale_paths = set(overlay.file_chunk_counts()) - (diff.added | diff.modified)
    if stale_paths:
        try:
            await overlay.delete_by_files(sorted(stale_paths))
        except Exception as exc:
            result.errors.append(f"Failed to prune stale overlay files: {exc}")

    return result, diff, layout
