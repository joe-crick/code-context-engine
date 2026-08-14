from __future__ import annotations

from pathlib import Path

import pytest

from context_engine.config import load_config
from context_engine.git import repository_storage_layout, resolve_git_repository_context
from context_engine.indexer.pipeline import run_indexing
from context_engine.indexer.worktree import sync_worktree_overlay
from context_engine.storage.local_backend import LocalBackend
from context_engine.storage.worktree_overlay import WorktreeOverlayBackend
from tests.test_git_repository_context import git, init_repo


class _StubEmbedder:
    cache_salt = "worktree-sync-test"
    dimension = 4

    def __init__(self, *args, **kwargs):
        self.cache = None

    def attach_cache(self, cache):
        self.cache = cache

    def embed(self, chunks, progress_fn=None):
        for chunk in chunks:
            chunk.embedding = [0.1, 0.2, 0.3, 0.4]
        if progress_fn:
            progress_fn(len(chunks), len(chunks))


@pytest.fixture(autouse=True)
def stub_embedder(monkeypatch):
    monkeypatch.setattr("context_engine.indexer.pipeline.Embedder", _StubEmbedder)

    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("context_engine.indexer.pipeline.asyncio.to_thread", inline_to_thread)


def _config(tmp_path):
    config = load_config()
    config.storage_path = str(tmp_path / "storage" / "projects")
    return config


async def _build_base(config, repo: Path):
    context = resolve_git_repository_context(repo)
    assert context is not None
    layout = repository_storage_layout(config, repo, context, migrate_legacy=False)
    result = await run_indexing(
        config,
        repo,
        full=True,
        storage_base_override=layout.base_dir,
    )
    assert not result.errors
    return layout


@pytest.mark.asyncio
async def test_sync_writes_only_worktree_delta_and_tracks_deletions(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "stable.py").write_text("def stable():\n    return 'base'\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "add stable")
    config = _config(tmp_path)
    await _build_base(config, repo)

    worktree = tmp_path / "repo-feature"
    git(repo, "worktree", "add", "-b", "feature", str(worktree))
    (worktree / "base.py").write_text("def base():\n    return 'worktree needle'\n")
    (worktree / "new.py").write_text("def new():\n    return 'new needle'\n")
    (worktree / "stable.py").unlink()

    result, diff, layout = await sync_worktree_overlay(config, worktree)
    overlay_files = LocalBackend(base_path=str(layout.worktree_dir)).file_chunk_counts()

    assert not result.errors
    assert set(result.indexed_files) == {"base.py", "new.py"}
    assert set(overlay_files) == {"base.py", "new.py"}
    assert diff.modified == {"base.py"}
    assert diff.added == {"new.py"}
    assert diff.deleted == {"stable.py"}

    second, second_diff, _ = await sync_worktree_overlay(config, worktree)
    assert second.indexed_files == []
    assert second_diff == diff


@pytest.mark.asyncio
async def test_sync_materializes_file_when_refreshed_base_advances(tmp_path):
    repo = init_repo(tmp_path / "repo")
    config = _config(tmp_path)
    await _build_base(config, repo)
    worktree = tmp_path / "repo-feature"
    git(repo, "worktree", "add", "-b", "feature", str(worktree))

    first, first_diff, layout = await sync_worktree_overlay(config, worktree)
    assert first.indexed_files == []
    assert not first_diff.modified
    assert LocalBackend(base_path=str(layout.worktree_dir)).count_chunks() == 0

    (repo / "base.py").write_text("def base():\n    return 'new shared base'\n")
    git(repo, "add", "base.py")
    git(repo, "commit", "-m", "advance base")
    await _build_base(config, repo)

    result, diff, layout = await sync_worktree_overlay(config, worktree)
    overlay = LocalBackend(base_path=str(layout.worktree_dir))
    backend = WorktreeOverlayBackend(
        base=LocalBackend(base_path=str(layout.base_dir)),
        overlay=overlay,
        diff=diff,
    )
    chunks = await backend.vector_search([0.1, 0.2, 0.3, 0.4], top_k=10)

    assert result.indexed_files == ["base.py"]
    assert diff.modified == {"base.py"}
    assert any("return 'base'" in chunk.content for chunk in chunks)
    assert all("new shared base" not in chunk.content for chunk in chunks)
