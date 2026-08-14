from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from context_engine.cli import main
from context_engine.config import Config
from context_engine.git import (
    repository_storage_layout,
    resolve_git_repository_context,
    resolve_repository_main_checkout,
)
from context_engine.git.diff import GitWorktreeDiff
from context_engine.models import Chunk, ChunkType
from context_engine.retrieval.retriever import HybridRetriever
from context_engine.storage.worktree_overlay import WorktreeOverlayBackend
from context_engine.utils import project_storage_dir, resolve_project_storage_dir
from tests.test_git_repository_context import git, init_repo


class _StubEmbedder:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]


class _MemoryBackend:
    def __init__(self):
        self.chunks: list[Chunk] = []

    async def ingest(self, chunks, nodes, edges):
        self.chunks.extend(chunks)

    async def vector_search(self, query_embedding, top_k=10, filters=None):
        chunks = self.chunks
        if filters and "file_path" in filters:
            chunks = [chunk for chunk in chunks if chunk.file_path == filters["file_path"]]
        return chunks[:top_k]

    async def fts_search(self, query, top_k=30):
        query = query.lower()
        return [
            (chunk.id, -rank)
            for rank, chunk in enumerate(self.chunks[:top_k])
            if query in chunk.content.lower()
        ]

    async def get_chunks_by_ids(self, chunk_ids):
        wanted = set(chunk_ids)
        return [chunk for chunk in self.chunks if chunk.id in wanted]

    async def get_chunk_by_id(self, chunk_id):
        return next((chunk for chunk in self.chunks if chunk.id == chunk_id), None)

    async def delete_by_file(self, file_path):
        self.chunks = [chunk for chunk in self.chunks if chunk.file_path != file_path]

    async def delete_by_files(self, file_paths):
        doomed = set(file_paths)
        self.chunks = [chunk for chunk in self.chunks if chunk.file_path not in doomed]

    def count_chunks(self):
        return len(self.chunks)


def _chunk(chunk_id: str, file_path: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        content=content,
        chunk_type=ChunkType.FUNCTION,
        file_path=file_path,
        start_line=1,
        end_line=2,
        language="python",
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


def _invoke_index_base(tmp_path, project_dir, monkeypatch, fake_run_indexing):
    cfg = Config(storage_path=str(tmp_path / "storage" / "projects"))
    monkeypatch.setattr("context_engine.indexer.pipeline.run_indexing", fake_run_indexing)
    monkeypatch.setattr("context_engine.cli._show_update_notice", lambda: None)
    with patch("context_engine.cli.load_config", return_value=cfg):
        result = CliRunner().invoke(
            main,
            ["index-base", "--project-dir", str(project_dir)],
        )
    return cfg, result


def test_index_base_from_main_checkout_writes_repository_base(tmp_path, monkeypatch):
    repo = init_repo(tmp_path / "repo")
    calls = []

    async def fake_run_indexing(config, project_dir, **kwargs):
        from context_engine.indexer.pipeline import IndexResult

        calls.append((Path(project_dir), kwargs))
        kwargs["progress_fn"](1, 1)
        kwargs["storage_base_override"].mkdir(parents=True)
        return IndexResult(indexed_files=["base.py"], total_chunks=1)

    cfg, result = _invoke_index_base(tmp_path, repo, monkeypatch, fake_run_indexing)
    context = resolve_git_repository_context(repo)
    assert context is not None
    layout = repository_storage_layout(cfg, repo, context, migrate_legacy=False)

    assert result.exit_code == 0, result.output
    assert calls[0][0] == repo.resolve()
    assert calls[0][1]["full"] is True
    assert calls[0][1]["storage_base_override"] == layout.base_dir
    assert f"Repository: {repo.resolve()}" in result.output
    assert f"Repo ID:    {context.repository_id}" in result.output
    assert f"Base store: {layout.base_dir}" in result.output
    assert "Indexing shared base..." in result.output
    assert "1/1 files" in result.output
    assert "Shared base index complete" in result.output
    assert "Files:  1" in result.output
    assert "Chunks: 1" in result.output
    assert f"Store:  {layout.base_dir}" in result.output


def test_index_base_from_linked_worktree_uses_same_main_checkout_and_base(
    tmp_path,
    monkeypatch,
):
    repo = init_repo(tmp_path / "repo")
    worktree = tmp_path / "repo-linked"
    git(repo, "worktree", "add", "-b", "feature", str(worktree))
    calls = []

    async def fake_run_indexing(config, project_dir, **kwargs):
        from context_engine.indexer.pipeline import IndexResult

        calls.append((Path(project_dir), kwargs))
        kwargs["storage_base_override"].mkdir(parents=True)
        return IndexResult(indexed_files=["base.py"], total_chunks=1)

    cfg, result = _invoke_index_base(tmp_path, worktree, monkeypatch, fake_run_indexing)
    repo_context = resolve_git_repository_context(repo)
    worktree_context = resolve_git_repository_context(worktree)
    assert repo_context is not None
    assert worktree_context is not None
    layout = repository_storage_layout(cfg, repo, repo_context, migrate_legacy=False)
    legacy_worktree = resolve_project_storage_dir(cfg, worktree)

    assert result.exit_code == 0, result.output
    assert repo_context.repository_id == worktree_context.repository_id
    assert resolve_repository_main_checkout(worktree) == repo.resolve()
    assert calls[0][0] == repo.resolve()
    assert calls[0][1]["full"] is True
    assert calls[0][1]["storage_base_override"] == layout.base_dir
    assert f"Repo ID:    {repo_context.repository_id}" in result.output
    assert f"Base store: {layout.base_dir}" in result.output
    assert not legacy_worktree.exists()
    assert not layout.worktree_dir.exists()


def test_index_base_main_file_is_retrievable_through_worktree_overlay(
    tmp_path,
    monkeypatch,
):
    repo = init_repo(tmp_path / "repo")
    (repo / "base.py").write_text("def base():\n    return 'baseneedle shared base'\n")
    worktree = tmp_path / "repo-linked"
    git(repo, "worktree", "add", "-b", "feature", str(worktree))
    base_backend = _MemoryBackend()

    async def fake_run_indexing(config, project_dir, **kwargs):
        from context_engine.indexer.pipeline import IndexResult

        await base_backend.ingest(
            [_chunk(
                "base-shared",
                "base.py",
                "def base():\n    return 'baseneedle shared base'\n",
            )],
            [],
            [],
        )
        return IndexResult(indexed_files=["base.py"], total_chunks=1)

    cfg, result = _invoke_index_base(tmp_path, worktree, monkeypatch, fake_run_indexing)
    assert result.exit_code == 0, result.output

    worktree_context = resolve_git_repository_context(worktree)
    assert worktree_context is not None
    layout = repository_storage_layout(cfg, worktree, worktree_context, migrate_legacy=False)
    assert f"Base store: {layout.base_dir}" in result.output
    backend = WorktreeOverlayBackend(
        base=base_backend,
        overlay=_MemoryBackend(),
        diff=GitWorktreeDiff(),
    )
    import asyncio

    results = asyncio.run(
        HybridRetriever(backend, _StubEmbedder()).retrieve(
            "baseneedle",
            top_k=5,
            confidence_threshold=0.0,
            marginal_ratio=0.0,
        )
    )

    assert any("baseneedle shared base" in chunk.content for chunk in results)


def test_index_base_outside_git_returns_useful_error(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()

    async def fake_run_indexing(*args, **kwargs):  # pragma: no cover
        raise AssertionError("indexing should not run outside Git")

    _, result = _invoke_index_base(tmp_path, outside, monkeypatch, fake_run_indexing)

    assert result.exit_code != 0
    assert "not a Git repository with a main checkout" in result.output


def test_index_base_failure_surfaces_repo_store_and_error(tmp_path, monkeypatch):
    repo = init_repo(tmp_path / "repo")

    async def fake_run_indexing(config, project_dir, **kwargs):
        from context_engine.indexer.pipeline import IndexResult

        return IndexResult(errors=["embed failed"])

    cfg, result = _invoke_index_base(tmp_path, repo, monkeypatch, fake_run_indexing)
    context = resolve_git_repository_context(repo)
    assert context is not None
    layout = repository_storage_layout(cfg, repo, context, migrate_legacy=False)

    assert result.exit_code != 0
    assert f"Repository: {repo.resolve()}" in result.output
    assert f"Base store: {layout.base_dir}" in result.output
    assert "embed failed" in result.output


def test_resolve_repository_main_checkout_preserves_non_worktree_checkout(tmp_path):
    repo = init_repo(tmp_path / "repo")

    assert resolve_repository_main_checkout(repo) == repo.resolve()


def test_existing_non_git_index_command_still_uses_project_storage(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    cfg = Config(storage_path=str(tmp_path / "storage" / "projects"))
    calls = []

    async def fake_run_indexing(config, project_dir, **kwargs):
        from context_engine.indexer.pipeline import IndexResult

        calls.append((Path(project_dir), kwargs))
        return IndexResult(indexed_files=["base.py"], total_chunks=1)

    monkeypatch.setattr("context_engine.indexer.pipeline.run_indexing", fake_run_indexing)
    monkeypatch.setattr("context_engine.cli._show_update_notice", lambda: None)
    with (
        patch("context_engine.cli.load_config", return_value=cfg),
        patch("context_engine.cli.Path.cwd", return_value=project),
    ):
        result = CliRunner().invoke(main, ["index", "--full"])

    assert result.exit_code == 0, result.output
    assert calls[0][0] == project.resolve()
    assert calls[0][1]["full"] is True
    assert calls[0][1]["storage_base_override"] == project_storage_dir(cfg, project)


def test_sync_worktree_reports_delta_and_overlay_path(tmp_path, monkeypatch):
    repo = init_repo(tmp_path / "repo")
    cfg = Config(storage_path=str(tmp_path / "storage" / "projects"))
    context = resolve_git_repository_context(repo)
    assert context is not None
    layout = repository_storage_layout(cfg, repo, context, migrate_legacy=False)

    async def fake_sync(config, project_dir, **kwargs):
        from context_engine.indexer.pipeline import IndexResult

        assert Path(project_dir) == repo.resolve()
        return (
            IndexResult(indexed_files=["changed.py"], total_chunks=3),
            GitWorktreeDiff(
                modified={"changed.py"},
                added={"new.py"},
                deleted={"gone.py"},
            ),
            layout,
        )

    monkeypatch.setattr("context_engine.indexer.worktree.sync_worktree_overlay", fake_sync)
    monkeypatch.setattr("context_engine.cli._show_update_notice", lambda: None)
    with patch("context_engine.cli.load_config", return_value=cfg):
        result = CliRunner().invoke(
            main,
            ["sync-worktree", "--project-dir", str(repo)],
        )

    assert result.exit_code == 0, result.output
    assert "Modified: 1" in result.output
    assert "Added:    1" in result.output
    assert "Deleted:  1" in result.output
    assert "Files indexed:  1" in result.output
    assert "Chunks indexed: 3" in result.output
    assert f"Overlay: {layout.worktree_dir}" in result.output
