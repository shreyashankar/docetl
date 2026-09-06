import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from server.app.models import PipelineConfigRequest, WorkspaceSaveRequest
from server.app.routes import filesystem


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DOCETL_HOME_DIR", str(home))
    root = home / ".docetl"
    root.mkdir()
    return root


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(filesystem.router, prefix="/fs")
    return TestClient(app)


def assert_bad_path(awaitable) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(awaitable)
    assert exc_info.value.status_code == 400


def test_raw_path_routes_reject_files_outside_storage_root(
    client: TestClient, storage_root: Path, tmp_path: Path
) -> None:
    outside_file = tmp_path / "outside.json"
    outside_file.write_text('{"safe_fixture": true}')

    for route in ("read-file", "read-file-page", "check-file"):
        response = client.get(f"/fs/{route}", params={"path": str(outside_file)})
        assert response.status_code == 400

    assert_bad_path(filesystem.serve_document(str(outside_file)))


def test_namespace_routes_reject_traversal(storage_root: Path, tmp_path: Path) -> None:
    outside_dir = storage_root.parent / "escaped"
    pipeline_request = PipelineConfigRequest(
        namespace="../escaped",
        name="pipeline",
        config="pipeline: {}",
        input_path="",
        output_path="",
    )

    assert_bad_path(filesystem.check_namespace("../escaped"))
    assert_bad_path(
        filesystem.upload_file(
            file=UploadFile(
                filename="data.json", file=BytesIO(b'{"safe_fixture": true}')
            ),
            url=None,
            namespace="../escaped",
        )
    )
    assert_bad_path(
        filesystem.save_documents(
            files=[UploadFile(filename="doc.txt", file=BytesIO(b"document"))],
            namespace="../escaped",
        )
    )
    assert_bad_path(filesystem.write_pipeline_config(pipeline_request))
    assert_bad_path(filesystem.load_workspace("../escaped"))
    assert_bad_path(
        filesystem.save_workspace(
            "../escaped", WorkspaceSaveRequest(content="workspace: {}")
        )
    )

    assert not outside_dir.exists()


def test_leaf_names_cannot_escape_their_directories(storage_root: Path) -> None:
    assert_bad_path(
        filesystem.upload_file(
            file=UploadFile(
                filename="../escaped.json",
                file=BytesIO(b'{"safe_fixture": true}'),
            ),
            url=None,
            namespace="workspace",
        )
    )

    assert_bad_path(
        filesystem.write_pipeline_config(
            PipelineConfigRequest(
                namespace="workspace",
                name="../escaped",
                config="pipeline: {}",
                input_path="",
                output_path="",
            )
        )
    )

    assert not (storage_root / "workspace" / "escaped.json").exists()
    assert not (storage_root / "workspace" / "escaped").exists()


def test_absolute_and_relative_paths_inside_storage_root_still_work(
    client: TestClient, storage_root: Path
) -> None:
    workspace = storage_root / "workspace"
    workspace.mkdir()
    data_file = workspace / "report..json"
    data_file.write_text('{"inside": true}')

    absolute_response = client.get("/fs/read-file", params={"path": str(data_file)})
    relative_response = client.get(
        "/fs/read-file", params={"path": "workspace/report..json"}
    )
    check_response = client.get("/fs/check-file", params={"path": str(data_file)})

    assert absolute_response.status_code == 200
    assert absolute_response.json() == {"inside": True}
    assert relative_response.status_code == 200
    assert relative_response.json() == {"inside": True}
    assert check_response.json() == {"exists": True}


def test_valid_write_routes_remain_available(storage_root: Path) -> None:
    absolute_namespace = storage_root / "team" / "workspace"

    namespace_result = asyncio.run(filesystem.check_namespace(str(absolute_namespace)))
    upload_result = asyncio.run(
        filesystem.upload_file(
            file=UploadFile(filename="data.json", file=BytesIO(b'{"inside": true}')),
            url=None,
            namespace="team/workspace",
        )
    )
    documents_result = asyncio.run(
        filesystem.save_documents(
            files=[UploadFile(filename="report..txt", file=BytesIO(b"document"))],
            namespace="team/workspace",
        )
    )
    pipeline_result = asyncio.run(
        filesystem.write_pipeline_config(
            PipelineConfigRequest(
                namespace="team/workspace",
                name="pipeline",
                config="pipeline: {}",
                input_path="",
                output_path="",
            )
        )
    )
    workspace_result = asyncio.run(
        filesystem.save_workspace(
            "team/workspace", WorkspaceSaveRequest(content="workspace: {}")
        )
    )

    assert namespace_result == {"exists": False}
    assert Path(upload_result["path"]).is_file()
    assert Path(documents_result["files"][0]["path"]).is_file()
    assert Path(pipeline_result["filePath"]).is_file()
    assert workspace_result == {"ok": True}
    assert asyncio.run(filesystem.load_workspace("team/workspace")) == {
        "content": "workspace: {}"
    }


def test_nested_workspace_routes_remain_available(
    client: TestClient, storage_root: Path
) -> None:
    save_response = client.post(
        "/fs/workspace/team/workspace", json={"content": "workspace: {}"}
    )
    load_response = client.get("/fs/workspace/team/workspace")

    assert save_response.status_code == 200
    assert save_response.json() == {"ok": True}
    assert load_response.status_code == 200
    assert load_response.json() == {"content": "workspace: {}"}
    assert (storage_root / "team" / "workspace" / "workspace.yaml").is_file()


def test_sibling_prefix_and_symlink_escapes_are_rejected(
    storage_root: Path, tmp_path: Path
) -> None:
    sibling = storage_root.parent / ".docetl-outside"
    sibling.mkdir()
    sibling_file = sibling / "data.json"
    sibling_file.write_text("{}")
    assert_bad_path(filesystem.read_file(str(sibling_file)))

    outside = tmp_path / "symlink-target"
    outside.mkdir()
    outside_file = outside / "data.json"
    outside_file.write_text("{}")
    link = storage_root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert_bad_path(filesystem.read_file(str(link / "data.json")))
    assert_bad_path(filesystem.check_namespace("link/created"))
    assert not (outside / "created").exists()


def test_workspace_leaf_symlink_escape_is_rejected(
    storage_root: Path, tmp_path: Path
) -> None:
    outside_file = tmp_path / "outside-workspace.yaml"
    outside_file.write_text("secret: fixture")
    workspace = storage_root / "workspace"
    workspace.mkdir()
    try:
        (workspace / "workspace.yaml").symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert_bad_path(filesystem.load_workspace("workspace"))


def test_malformed_paths_return_bad_request(
    client: TestClient, storage_root: Path
) -> None:
    for route in ("read-file", "read-file-page", "check-file"):
        response = client.get(f"/fs/{route}", params={"path": "bad\x00path"})
        assert response.status_code == 400


def test_missing_inside_paths_preserve_not_found_status(storage_root: Path) -> None:
    with pytest.raises(HTTPException) as read_error:
        asyncio.run(filesystem.read_file(str(storage_root / "missing.json")))
    assert read_error.value.status_code == 404

    with pytest.raises(HTTPException) as workspace_error:
        asyncio.run(filesystem.load_workspace("missing"))
    assert workspace_error.value.status_code == 404
