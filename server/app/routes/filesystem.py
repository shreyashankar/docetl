import csv
import json
import os
import shutil
from io import StringIO
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from server.app.models import PipelineConfigRequest, WorkspaceSaveRequest

router = APIRouter()


def get_home_dir() -> str:
    """Get the home directory from env var or user home"""
    return os.getenv("DOCETL_HOME_DIR", os.path.expanduser("~"))


def get_storage_root() -> Path:
    """Return the canonical root for all server-managed files."""
    return (Path(get_home_dir()).expanduser() / ".docetl").resolve()


def resolve_storage_path(path: str | Path) -> Path:
    """Resolve a path and reject targets outside the server storage root."""
    root = get_storage_root()
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Path escapes the DocETL storage root"
        )
    return resolved


def resolve_storage_child(parent: Path, name: str, label: str) -> Path:
    """Resolve a single path component beneath a storage directory."""
    if (
        not name
        or name in {".", ".."}
        or Path(name).is_absolute()
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return resolve_storage_path(parent / name)


def get_namespace_dir(namespace: str) -> Path:
    """Get the namespace directory path"""
    if not namespace:
        raise HTTPException(status_code=400, detail="Invalid namespace")
    return resolve_storage_path(namespace)


@router.post("/check-namespace")
async def check_namespace(namespace: str):
    """Check if namespace exists and create if it doesn't"""
    try:
        namespace_dir = get_namespace_dir(namespace)
        exists = namespace_dir.exists()

        if not exists:
            namespace_dir.mkdir(parents=True, exist_ok=True)

        return {"exists": exists}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check/create namespace: {str(e)}"
        )


def validate_json_content(content: bytes) -> None:
    """Validate that content can be parsed as JSON"""
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")


def convert_csv_to_json(csv_content: bytes) -> bytes:
    """Convert CSV content to JSON format"""
    try:
        # Decode bytes to string and create a StringIO object
        csv_string = csv_content.decode("utf-8")
        csv_file = StringIO(csv_string)

        # Read CSV and convert to list of dictionaries
        reader = csv.DictReader(csv_file)
        data = list(reader)

        if not data:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Convert back to JSON bytes
        return json.dumps(data).encode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CSV encoding")
    except csv.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")


def is_likely_csv(content: bytes, filename: str) -> bool:
    """Check if content is likely to be CSV based on content and filename"""
    # Check filename extension
    if filename.lower().endswith(".csv"):
        return True

    # If no clear extension, try to detect CSV content
    try:
        # Take first line and check if it looks like CSV
        first_line = content.split(b"\n")[0].decode("utf-8")
        # Check if line contains commas and no obvious JSON characters
        return "," in first_line and not any(c in first_line for c in "{}[]")
    except UnicodeDecodeError:
        return False


@router.post("/upload-file")
async def upload_file(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    namespace: str = Form(...),
):
    """Upload a file to the namespace files directory, either from a direct upload or a URL"""
    try:
        if not file and not url:
            raise HTTPException(
                status_code=400, detail="Either file or url must be provided"
            )

        upload_dir = resolve_storage_path(get_namespace_dir(namespace) / "files")
        upload_dir.mkdir(parents=True, exist_ok=True)

        if url:
            # Get filename from URL or default to dataset.json
            filename = url.split("/")[-1] or "dataset.json"

            file_path = resolve_storage_child(
                upload_dir, filename.replace(".csv", ".json"), "filename"
            )

            # Handle URL download
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET",
                    url,
                    follow_redirects=True,
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download from URL: {response.status_code}",
                        )

                    # Save the file in chunks
                    content_chunks = []
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            content_chunks.append(chunk)

                    # Combine chunks
                    content = b"".join(content_chunks)

                    # Check if content is CSV and convert if needed
                    if is_likely_csv(content, filename):
                        try:
                            content = convert_csv_to_json(content)
                        except HTTPException as e:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Failed to convert CSV to JSON: {str(e.detail)}",
                            )

                    # Validate JSON content
                    validate_json_content(content)

                    # Write to file
                    with file_path.open("wb") as f:
                        f.write(content)
        else:
            # Handle direct file upload
            if not file.filename:
                raise HTTPException(status_code=400, detail="Invalid filename")
            file_content = await file.read()

            # Check if content is CSV and convert if needed
            if file.filename.lower().endswith(".csv"):
                try:
                    file_content = convert_csv_to_json(file_content)
                except HTTPException as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to convert CSV to JSON: {str(e.detail)}",
                    )

            # Validate JSON content
            validate_json_content(file_content)

            # Always save as .json
            file_path = resolve_storage_child(
                upload_dir, file.filename.replace(".csv", ".json"), "filename"
            )
            with file_path.open("wb") as f:
                f.write(file_content)

        return {"path": str(file_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/save-documents")
async def save_documents(
    files: list[UploadFile] = File(...), namespace: str = Form(...)
):
    """Save multiple documents to the namespace documents directory"""
    try:
        uploads_dir = resolve_storage_path(get_namespace_dir(namespace) / "documents")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for file in files:
            # Create safe filename
            safe_name = "".join(
                c if c.isalnum() or c in ".-" else "_" for c in file.filename
            )
            file_path = resolve_storage_child(uploads_dir, safe_name, "filename")

            with file_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            saved_files.append({"name": file.filename, "path": str(file_path)})

        return {"files": saved_files}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save documents: {str(e)}"
        )


@router.post("/write-pipeline-config")
async def write_pipeline_config(request: PipelineConfigRequest):
    """Write pipeline configuration YAML file"""
    try:
        pipeline_dir = resolve_storage_path(
            get_namespace_dir(request.namespace) / "pipelines"
        )
        config_dir = resolve_storage_path(pipeline_dir / "configs")
        pipeline_name = resolve_storage_child(
            pipeline_dir, request.name, "pipeline name"
        )
        name_dir = resolve_storage_path(pipeline_name / "intermediates")

        config_dir.mkdir(parents=True, exist_ok=True)
        name_dir.mkdir(parents=True, exist_ok=True)

        file_path = resolve_storage_child(
            config_dir, f"{request.name}.yaml", "pipeline filename"
        )
        with file_path.open("w") as f:
            f.write(request.config)

        return {
            "filePath": str(file_path),
            "inputPath": request.input_path,
            "outputPath": request.output_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write pipeline configuration: {str(e)}"
        )


@router.get("/read-file")
async def read_file(path: str):
    """Read file contents"""
    try:
        if path.startswith(("http://", "https://")):
            # For HTTP URLs, we'll need to implement request handling
            raise HTTPException(
                status_code=400, detail="HTTP URLs not supported in this endpoint"
            )

        file_path = resolve_storage_path(path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/read-file-page")
async def read_file_page(path: str, page: int = 0, chunk_size: int = 500000):
    """Read file contents by page"""
    try:
        file_path = resolve_storage_path(path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        file_size = file_path.stat().st_size
        start = page * chunk_size

        with file_path.open("rb") as f:
            f.seek(start)
            content = f.read(chunk_size).decode("utf-8")

        return {
            "content": content,
            "totalSize": file_size,
            "page": page,
            "hasMore": start + len(content) < file_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/serve-document/{path:path}")
async def serve_document(path: str):
    """Serve document files"""
    try:
        file_path = resolve_storage_path(path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path,
            filename=file_path.name,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


@router.get("/workspace/{workspace_id:path}")
async def load_workspace(workspace_id: str):
    """Load workspace state YAML for a given workspace UUID"""
    try:
        workspace_file = resolve_storage_path(
            get_namespace_dir(workspace_id) / "workspace.yaml"
        )
        if not workspace_file.is_file():
            raise HTTPException(status_code=404, detail="Workspace not found")
        with workspace_file.open("r") as f:
            content = f.read()
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load workspace: {str(e)}"
        )


@router.post("/workspace/{workspace_id:path}")
async def save_workspace(workspace_id: str, request: WorkspaceSaveRequest):
    """Save workspace state YAML for a given workspace UUID"""
    try:
        namespace_dir = get_namespace_dir(workspace_id)
        namespace_dir.mkdir(parents=True, exist_ok=True)
        workspace_file = resolve_storage_path(namespace_dir / "workspace.yaml")
        with workspace_file.open("w") as f:
            f.write(request.content)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save workspace: {str(e)}"
        )


@router.get("/check-file")
async def check_file(path: str):
    """Check if a file exists without reading it"""
    try:
        file_path = resolve_storage_path(path)
        exists = file_path.exists()
        return {"exists": exists}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check file: {str(e)}")
