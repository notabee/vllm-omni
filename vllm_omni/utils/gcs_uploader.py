# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Google Cloud Storage (GCS) upload utility for multimodal pipeline artifacts and logs.

Supports uploading generated multimodal outputs (.txt, .wav) to gs://<bucket>/outputs/
and container execution logs (.log) to gs://<bucket>/logs/ with automatic MIME type
detection, upload integrity verification, and bucket validation.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("vllm_omni.utils.gcs_uploader")

try:
    from google.cloud import storage
    _HAS_GCS = True
except ImportError:
    storage = None  # type: ignore[assignment]
    _HAS_GCS = False

DEFAULT_BUCKET_NAME = "rishabh-speechmaxxxing"
DEFAULT_EXPECTED_LOCATION = "US-CENTRAL1"
DEFAULT_PROJECT_FALLBACK = "cloud-tpu-shared-capacity"

MIME_TYPE_OVERRIDES: Dict[str, str] = {
    ".wav": "audio/wav",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".csv": "text/csv",
    ".bin": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".safetensors": "application/octet-stream",
}


def get_content_type(file_path: Union[str, Path]) -> str:
    """Determine MIME content-type for a file path.

    Args:
        file_path: Local file path or filename.

    Returns:
        MIME type string (e.g. 'audio/wav', 'text/plain').
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in MIME_TYPE_OVERRIDES:
        return MIME_TYPE_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def get_storage_client(
    project: Optional[str] = None,
    credentials: Any = None,
) -> storage.Client:
    """Initialize a Google Cloud Storage client with robust project fallback.

    Args:
        project: Optional GCP project ID. If not provided, inspects
            GOOGLE_CLOUD_PROJECT / GCP_PROJECT environment variables before
            falling back to default configuration or 'cloud-tpu-shared-capacity'.
        credentials: Optional Google Auth credentials object.

    Returns:
        google.cloud.storage.Client instance.

    Raises:
        ImportError: If google-cloud-storage package is not installed.
    """
    if storage is None:
        raise ImportError(
            "google-cloud-storage package is required for GCS operations. "
            "Please install it via `pip install google-cloud-storage`."
        )

    resolved_project = (
        project
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )

    if credentials is not None:
        return storage.Client(project=resolved_project, credentials=credentials)

    if resolved_project:
        return storage.Client(project=resolved_project)

    try:
        return storage.Client()
    except Exception as exc:
        logger.warning(
            "Default GCS client initialization returned '%s'. "
            "Falling back to project '%s'.",
            exc,
            DEFAULT_PROJECT_FALLBACK,
        )
        return storage.Client(project=DEFAULT_PROJECT_FALLBACK)


def verify_bucket(
    bucket_name: str = DEFAULT_BUCKET_NAME,
    client: Optional[storage.Client] = None,
    expected_location: Optional[str] = DEFAULT_EXPECTED_LOCATION,
) -> Dict[str, Any]:
    """Verify that a GCS bucket exists and is located in the expected region.

    Args:
        bucket_name: Name of the GCS bucket.
        client: Optional GCS client instance.
        expected_location: Expected region (default: US-CENTRAL1).

    Returns:
        Dictionary containing bucket metadata (exists, name, location,
        storage_class, uniform_bucket_level_access, storage_url).

    Raises:
        RuntimeError: If bucket does not exist or is inaccessible.
        ValueError: If bucket region does not match expected_location.
    """
    if client is None:
        client = get_storage_client()

    try:
        bucket = client.get_bucket(bucket_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to access GCS bucket '{bucket_name}': {exc}"
        ) from exc

    location = str(bucket.location).upper()
    storage_class = str(bucket.storage_class)

    ubla = False
    try:
        ubla = bool(getattr(bucket.iam_configuration, "uniform_bucket_level_access_enabled", False))
    except Exception:
        pass

    if expected_location:
        expected_norm = expected_location.strip().upper()
        if location != expected_norm:
            raise ValueError(
                f"Bucket '{bucket_name}' location mismatch: expected '{expected_norm}', "
                f"found '{location}'."
            )

    return {
        "exists": True,
        "name": bucket_name,
        "location": location,
        "storage_class": storage_class,
        "uniform_bucket_level_access": ubla,
        "storage_url": f"gs://{bucket_name}/",
    }


def upload_file_to_gcs(
    bucket_name: str,
    local_file_path: Union[str, Path],
    gcs_blob_name: str,
    client: Optional[storage.Client] = None,
    content_type: Optional[str] = None,
    verify_upload: bool = True,
) -> str:
    """Upload a single local file to a GCS bucket.

    Args:
        bucket_name: Target GCS bucket name.
        local_file_path: Path to the local file to upload.
        gcs_blob_name: Destination object key in GCS.
        client: Optional GCS client instance.
        content_type: Optional explicit MIME content type. If None, inferred
            from file extension.
        verify_upload: If True, reloads blob metadata to confirm upload succeeded.

    Returns:
        Full GCS URI of the uploaded object (e.g. 'gs://bucket/path/file.txt').

    Raises:
        FileNotFoundError: If local_file_path does not exist or is not a file.
        RuntimeError: If upload verification fails.
    """
    path = Path(local_file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local file does not exist: {path}")

    if client is None:
        client = get_storage_client()

    bucket = client.bucket(bucket_name)
    clean_blob_name = gcs_blob_name.lstrip("/")
    blob = bucket.blob(clean_blob_name)

    resolved_content_type = content_type or get_content_type(path)
    file_size = path.stat().st_size

    logger.info(
        "[GCS Upload] Uploading %s (%d bytes) -> gs://%s/%s [%s]",
        path.name,
        file_size,
        bucket_name,
        clean_blob_name,
        resolved_content_type,
    )

    blob.upload_from_filename(str(path), content_type=resolved_content_type)

    if verify_upload:
        blob.reload()
        if blob.size is None:
            raise RuntimeError(
                f"Upload verification failed: blob gs://{bucket_name}/{clean_blob_name} "
                "not found after upload."
            )
        if blob.size != file_size:
            raise RuntimeError(
                f"Upload byte integrity mismatch for gs://{bucket_name}/{clean_blob_name}: "
                f"expected {file_size} bytes, GCS reported {blob.size} bytes."
            )

    uri = f"gs://{bucket_name}/{clean_blob_name}"
    return uri


def upload_directory_to_gcs(
    bucket_name: str,
    local_dir: Union[str, Path],
    gcs_prefix: str = "",
    client: Optional[storage.Client] = None,
    allowed_extensions: Optional[Sequence[str]] = None,
    recursive: bool = True,
    verify_upload: bool = True,
) -> List[str]:
    """Upload files from a local directory to a GCS bucket prefix.

    Args:
        bucket_name: Target GCS bucket name.
        local_dir: Local directory path containing files.
        gcs_prefix: Prefix in GCS (e.g. 'outputs' or 'logs').
        client: Optional GCS client instance.
        allowed_extensions: Optional list of allowed suffixes (e.g. ['.txt', '.wav']).
            If None, all files are uploaded.
        recursive: Whether to scan subdirectories recursively.
        verify_upload: Whether to verify uploaded blob size.

    Returns:
        List of uploaded object GCS URIs.
    """
    local_path = Path(local_dir).resolve()
    if not local_path.exists():
        logger.warning(
            "[GCS Upload] Directory %s does not exist. Skipping upload.",
            local_dir,
        )
        return []

    if client is None:
        client = get_storage_client()

    allowed_ext_set = (
        {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions}
        if allowed_extensions
        else None
    )

    clean_prefix = gcs_prefix.strip("/")
    uploaded_uris: List[str] = []

    pattern = "**/*" if recursive else "*"
    for file_path in sorted(local_path.glob(pattern)):
        if not file_path.is_file():
            continue

        if allowed_ext_set and file_path.suffix.lower() not in allowed_ext_set:
            continue

        rel_path = file_path.relative_to(local_path)
        blob_name = f"{clean_prefix}/{rel_path}" if clean_prefix else str(rel_path)

        uri = upload_file_to_gcs(
            bucket_name=bucket_name,
            local_file_path=file_path,
            gcs_blob_name=blob_name,
            client=client,
            verify_upload=verify_upload,
        )
        uploaded_uris.append(uri)

    return uploaded_uris


def upload_pipeline_artifacts(
    bucket_name: str = DEFAULT_BUCKET_NAME,
    output_dir: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[Union[str, Path]] = None,
    project: Optional[str] = None,
    verify_bucket_location: bool = True,
) -> Dict[str, List[str]]:
    """Upload inference outputs and logs to their designated GCS prefixes.

    Inference outputs (.txt, .wav) are uploaded to gs://<bucket>/outputs/.
    Execution logs (.log, .txt) are uploaded to gs://<bucket>/logs/.

    Args:
        bucket_name: GCS bucket name (default: 'rishabh-speechmaxxxing').
        output_dir: Local directory containing outputs (e.g. /workspace/output_audio).
        log_dir: Local directory containing logs (e.g. /workspace/logs).
        log_file: Optional single log file path (e.g. /workspace/logs/job_execution.log).
        project: Optional GCP project ID.
        verify_bucket_location: Whether to assert bucket exists and is in us-central1.

    Returns:
        Dictionary mapping prefix category ('outputs', 'logs') to list of uploaded GCS URIs.
    """
    client = get_storage_client(project=project)

    if verify_bucket_location:
        meta = verify_bucket(
            bucket_name=bucket_name,
            client=client,
            expected_location=DEFAULT_EXPECTED_LOCATION,
        )
        logger.info(
            "[GCS Upload] Verified bucket %s (%s, %s)",
            meta["name"],
            meta["location"],
            meta["storage_class"],
        )

    results: Dict[str, List[str]] = {"outputs": [], "logs": []}

    # 1. Upload outputs (.txt, .wav)
    if output_dir:
        out_path = Path(output_dir)
        if out_path.exists():
            uploaded_outputs = upload_directory_to_gcs(
                bucket_name=bucket_name,
                local_dir=out_path,
                gcs_prefix="outputs",
                client=client,
                allowed_extensions=[".txt", ".wav", ".json"],
            )
            results["outputs"].extend(uploaded_outputs)
        else:
            logger.warning("[GCS Upload] Output dir %s not found.", output_dir)

    # 2. Upload log directory
    if log_dir:
        l_path = Path(log_dir)
        if l_path.exists():
            uploaded_logs = upload_directory_to_gcs(
                bucket_name=bucket_name,
                local_dir=l_path,
                gcs_prefix="logs",
                client=client,
                allowed_extensions=[".log", ".txt"],
            )
            results["logs"].extend(uploaded_logs)
        else:
            logger.warning("[GCS Upload] Log dir %s not found.", log_dir)

    # 3. Upload single log file if specified and not already uploaded
    if log_file:
        lf_path = Path(log_file).resolve()
        if lf_path.is_file():
            blob_name = f"logs/{lf_path.name}"
            uri = upload_file_to_gcs(
                bucket_name=bucket_name,
                local_file_path=lf_path,
                gcs_blob_name=blob_name,
                client=client,
            )
            if uri not in results["logs"]:
                results["logs"].append(uri)
        else:
            logger.warning("[GCS Upload] Log file %s not found.", log_file)

    logger.info(
        "[GCS Upload] Finished pipeline upload: %d outputs, %d logs.",
        len(results["outputs"]),
        len(results["logs"]),
    )
    return results


def download_file_from_gcs(
    bucket_name: str,
    gcs_blob_name: str,
    local_dest_path: Union[str, Path],
    client: Optional[storage.Client] = None,
) -> Path:
    """Download an object from GCS to a local file.

    Args:
        bucket_name: Target GCS bucket name.
        gcs_blob_name: GCS object key.
        local_dest_path: Local destination file path.
        client: Optional GCS client instance.

    Returns:
        Path to downloaded local file.
    """
    dest = Path(local_dest_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = get_storage_client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name.lstrip("/"))

    if not blob.exists():
        raise FileNotFoundError(
            f"GCS object gs://{bucket_name}/{gcs_blob_name} does not exist."
        )

    blob.download_to_filename(str(dest))
    return dest


def delete_gcs_blob(
    bucket_name: str,
    gcs_blob_name: str,
    client: Optional[storage.Client] = None,
) -> bool:
    """Delete a single object from a GCS bucket.

    Args:
        bucket_name: Target GCS bucket name.
        gcs_blob_name: GCS object key to delete.
        client: Optional GCS client instance.

    Returns:
        True if deleted or object did not exist, False on failure.
    """
    if client is None:
        client = get_storage_client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name.lstrip("/"))

    if blob.exists():
        blob.delete()
        logger.info("[GCS Upload] Deleted gs://%s/%s", bucket_name, gcs_blob_name)
        return True
    return True


def delete_gcs_prefix(
    bucket_name: str,
    gcs_prefix: str,
    client: Optional[storage.Client] = None,
) -> int:
    """Delete all objects matching a GCS prefix.

    Args:
        bucket_name: Target GCS bucket name.
        gcs_prefix: Prefix to match (e.g. 'test_probes/').
        client: Optional GCS client instance.

    Returns:
        Count of deleted objects.
    """
    if client is None:
        client = get_storage_client()

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix.lstrip("/")))
    deleted_count = 0

    for blob in blobs:
        blob.delete()
        deleted_count += 1

    logger.info(
        "[GCS Upload] Deleted %d objects matching prefix 'gs://%s/%s'",
        deleted_count,
        bucket_name,
        gcs_prefix,
    )
    return deleted_count


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload inference artifacts and logs to Google Cloud Storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET_NAME,
        help="Target GCS bucket name.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional GCP project ID.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory containing outputs (.txt, .wav) to upload to outputs/.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Local directory containing logs (.log) to upload to logs/.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional single log file to upload to logs/.",
    )
    parser.add_argument(
        "--verify-bucket",
        action="store_true",
        help="Verify bucket existence and region before uploading.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for GCS uploader utility."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        res = upload_pipeline_artifacts(
            bucket_name=args.bucket,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            log_file=args.log_file,
            project=args.project,
            verify_bucket_location=args.verify_bucket or True,
        )
        print(f"[SUCCESS] Uploaded {len(res['outputs'])} output artifacts and {len(res['logs'])} log artifacts.")
        for out_uri in res["outputs"]:
            print(f"  - Output: {out_uri}")
        for log_uri in res["logs"]:
            print(f"  - Log: {log_uri}")
        return 0
    except Exception as exc:
        logger.error("[FAILED] GCS upload encountered an error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
