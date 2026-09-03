"""Timeline-aware finished-video save and load nodes.

The live sampler preview deliberately keeps its short-lived WebP images in
memory.  Finished renders need a much smaller durable representation: their
real encoded media plus the structural timeline that describes source shots,
serial chunks, and the exact Gemma description used for each chunk.

This module stores that timeline in two places for ordinary video formats:
the selected native ComfyUI or Video Helper Suite encoder writes container
metadata, and we always write a neighbouring JSON sidecar. The sidecar is the
authoritative fallback because some transcoders remove arbitrary container
tags. EXR is an image sequence, so its sidecar is primary; the first EXR also
receives the timeline header tag when ComfyUI's OpenEXR helper is available.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import shutil
import sys
import threading
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlencode

import av
import numpy as np
import torch
from aiohttp import web

import folder_paths
from comfy_api.latest import InputImpl, Types, io

try:
    from server import PromptServer
except ImportError:  # pragma: no cover - exercised by the lightweight tests.
    PromptServer = None


HREndlessTimeline = io.Custom("HRENDLESS_TIMELINE")

TIMELINE_SCHEMA_VERSION = 1
TIMELINE_TAG = "hr_endless_sampler_timeline"
SIDECAR_SUFFIX = ".hr_endless_sampler_timeline.json"
PLAYER_EVENT = "hr_endless_sampler_saved_video"
PLAYER_CACHE_LIMIT = 16
OUTPUT_BROWSER_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".gif", ".webp"}
OUTPUT_UPLOAD_VIDEO_EXTENSIONS = OUTPUT_BROWSER_VIDEO_EXTENSIONS - {".gif", ".webp"}
OUTPUT_UPLOAD_SUBFOLDER = "hr_endless_sampler_uploads"
OUTPUT_UPLOAD_CHUNK_DIRECTORY = "hr_endless_sampler_video_upload_chunks"
_SAFE_UPLOAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_SAFE_UPLOAD_NAME = re.compile(r"[^A-Za-z0-9._()\- ]+")

_PLAYER_CACHE: OrderedDict[str, dict] = OrderedDict()
_PLAYER_CACHE_LOCK = threading.Lock()


def _output_browser_directory(relative_path: str = "") -> tuple[Path, Path, str]:
    """Resolve one browser directory without permitting output-root escape."""
    root = Path(folder_paths.get_output_directory()).resolve()
    raw = str(relative_path or "").strip().replace("\\", "/")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError("output browser path must be relative")
    candidate = (root / relative).resolve()
    try:
        normalized = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("output browser path leaves ComfyUI's output directory") from error
    if not candidate.is_dir():
        raise ValueError(f"output folder does not exist: {raw or '/'}")
    normalized_text = "" if normalized == Path(".") else normalized.as_posix()
    return root, candidate, normalized_text


def _output_browser_listing(relative_path: str = "") -> dict:
    """List navigable folders and supported finished media in output/."""
    root, directory, normalized = _output_browser_directory(relative_path)
    entries: list[dict] = []
    sequence_frames: set[str] = set()

    for path in directory.iterdir():
        if not path.is_file() or not path.name.endswith(SIDECAR_SUFFIX):
            continue
        _timeline, media = _read_sidecar(path)
        if not isinstance(media, dict) or media.get("kind") != "exr_sequence":
            continue
        frames = media.get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        sequence_frames.update(str(item) for item in frames)
        stat = path.stat()
        entries.append({
            "kind": "sequence",
            "name": f"{frames[0]}  ({len(frames)}-frame EXR sequence)",
            "path": path.relative_to(root).as_posix(),
            "size": int(sum((directory / str(item)).stat().st_size for item in frames if (directory / str(item)).is_file())),
            "modified": float(stat.st_mtime),
            "frames": len(frames),
        })

    for path in directory.iterdir():
        try:
            stat = path.stat()
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"kind": "directory", "name": path.name, "path": relative, "size": 0, "modified": float(stat.st_mtime)})
        elif not path.is_file() or path.name.endswith(SIDECAR_SUFFIX):
            continue
        elif path.suffix.lower() in OUTPUT_BROWSER_VIDEO_EXTENSIONS:
            entries.append({"kind": "video", "name": path.name, "path": relative, "size": int(stat.st_size), "modified": float(stat.st_mtime)})
        elif path.suffix.lower() == ".exr" and path.name not in sequence_frames:
            entries.append({"kind": "exr", "name": path.name, "path": relative, "size": int(stat.st_size), "modified": float(stat.st_mtime)})

    entries.sort(key=lambda item: (item["kind"] != "directory", str(item["name"]).casefold()))
    parent = "" if not normalized else Path(normalized).parent.as_posix()
    if parent == ".":
        parent = ""
    return {"path": normalized, "parent": parent, "entries": entries}


def _matching_output_listing(value: str, *, strip_counter: bool = False) -> dict:
    """List output media beginning with a Save prefix or a loaded file's prefix."""
    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(Path(folder_paths.get_output_directory()).resolve())
        except ValueError as error:
            raise ValueError("matching-video path must be inside ComfyUI's output directory") from error
    parent = "" if path.parent == Path(".") else path.parent.as_posix()
    name = path.name
    if strip_counter:
        counter_match = re.match(r"^(.*)_\d+_.*$", name)
        if counter_match:
            name = counter_match.group(1)
        else:
            name = Path(name).stem
    elif Path(name).suffix.lower() in OUTPUT_BROWSER_VIDEO_EXTENSIONS | {".exr"}:
        name = Path(name).stem
    listing = _output_browser_listing(parent)
    entries = [
        item for item in listing["entries"]
        if item["kind"] != "directory" and Path(str(item["path"])).name.startswith(name)
    ]
    entries.sort(key=lambda item: (-float(item.get("modified") or 0.0), str(item.get("name") or "").casefold()))
    normalized_prefix = f"{parent}/{name}" if parent else name
    return {"prefix": normalized_prefix, "entries": entries}


def _safe_upload_filename(value: str) -> str:
    name = Path(str(value or "video.mp4").replace("\\", "/")).name
    name = _SAFE_UPLOAD_NAME.sub("_", name).strip(" ._")
    if not name:
        name = "video.mp4"
    if Path(name).suffix.lower() not in OUTPUT_UPLOAD_VIDEO_EXTENSIONS:
        raise ValueError("uploaded file must be a supported video")
    return name


def _unique_uploaded_video_path(filename: str) -> Path:
    directory = Path(folder_paths.get_output_directory()) / OUTPUT_UPLOAD_SUBFOLDER
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _copy_json(value):
    """Return a JSON-only deep copy, rejecting non-serializable timeline data."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def normalize_timeline(timeline, *, fps: float, total_frames: int) -> dict:
    """Validate and normalize the small timeline value passed between nodes."""
    source = timeline if isinstance(timeline, dict) else {}
    resolved_fps = _number(source.get("fps"), _number(fps, 24.0))
    if resolved_fps <= 0:
        resolved_fps = 24.0
    resolved_total = int(source.get("total_frames") or total_frames or 0)
    if resolved_total <= 0:
        resolved_total = max(0, int(total_frames))

    chunks = []
    for index, item in enumerate(source.get("chunks", ())):
        if not isinstance(item, dict):
            continue
        start = max(0, int(item.get("start", 0)))
        end = min(resolved_total - 1, int(item.get("end", start)))
        if end < start:
            continue
        chunk = {
            "chunk": int(item.get("chunk", index + 1)),
            "start": start,
            "end": end,
        }
        description = item.get("gemma_detailed_description")
        if isinstance(description, str) and description.strip():
            chunk["gemma_detailed_description"] = description.strip()
        for key in (
            "h3_render_seconds",
            "gemma_seconds",
            "gemma_preproduction_seconds",
            "chunk_total_seconds",
        ):
            value = _number(item.get(key), -1.0)
            if value >= 0:
                chunk[key] = value
        chunks.append(chunk)
    if not chunks and resolved_total:
        chunks.append({"chunk": 1, "start": 0, "end": resolved_total - 1})

    shots = []
    for index, item in enumerate(source.get("shots", ())):
        if not isinstance(item, dict):
            continue
        start = max(0, int(item.get("start", 0)))
        end = min(resolved_total - 1, int(item.get("end", start)))
        if end < start:
            continue
        shots.append({
            "shot": int(item.get("shot", index + 1)),
            "start": start,
            "end": end,
            "source_end": int(item.get("source_end", end)),
        })

    normalized = {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "producer": "HR Endless Sampler",
        "fps": resolved_fps,
        "total_frames": resolved_total,
        "chunks": chunks,
        "shots": shots,
    }
    render_total_seconds = _number(source.get("render_total_seconds"), -1.0)
    if render_total_seconds >= 0:
        normalized["render_total_seconds"] = render_total_seconds
    return _copy_json(normalized)


def _timeline_sidecar_path(media_path: Path) -> Path:
    return Path(str(media_path) + SIDECAR_SUFFIX)


def _write_sidecar(media_path: Path, timeline: dict, media: dict) -> Path:
    sidecar = _timeline_sidecar_path(media_path)
    payload = _copy_json({
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline": timeline,
        "media": media,
    })
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)
    return sidecar


def _read_sidecar(media_path: Path) -> tuple[dict | None, dict | None]:
    candidates = [media_path] if media_path.name.endswith(SIDECAR_SUFFIX) else [_timeline_sidecar_path(media_path)]
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("timeline"), dict):
            return payload["timeline"], payload.get("media") if isinstance(payload.get("media"), dict) else None
    return None, None


def _read_embedded_timeline(media_path: Path) -> dict | None:
    if media_path.suffix.lower() == ".exr":
        try:
            raw = media_path.read_bytes()
            if raw[:4] == b"\x76\x2f\x31\x01":
                position = 8
                while position < len(raw) and raw[position] != 0:
                    name_end = raw.index(b"\x00", position)
                    type_end = raw.index(b"\x00", name_end + 1)
                    size_start = type_end + 1
                    size = int.from_bytes(raw[size_start:size_start + 4], "little", signed=True)
                    value_start = size_start + 4
                    name = raw[position:name_end].decode("utf-8", errors="replace")
                    kind = raw[name_end + 1:type_end].decode("ascii", errors="replace")
                    if name == TIMELINE_TAG and kind == "string":
                        payload = json.loads(raw[value_start:value_start + size].decode("utf-8"))
                        return payload if isinstance(payload, dict) else None
                    position = value_start + size
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    try:
        with av.open(str(media_path)) as container:
            metadata = dict(container.metadata or {})
    except Exception:
        return None
    raw = metadata.get(TIMELINE_TAG)
    if raw is None:
        # FFmpeg/container combinations may normalize tag case.
        raw = next((value for key, value in metadata.items() if key.lower() == TIMELINE_TAG), None)
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _store_player_state(node_id, payload: dict):
    if node_id is None:
        return
    key = str(node_id)
    with _PLAYER_CACHE_LOCK:
        _PLAYER_CACHE[key] = _copy_json(payload)
        _PLAYER_CACHE.move_to_end(key)
        while len(_PLAYER_CACHE) > PLAYER_CACHE_LIMIT:
            _PLAYER_CACHE.popitem(last=False)


def _player_state(node_id) -> dict | None:
    key = str(node_id)
    with _PLAYER_CACHE_LOCK:
        payload = _PLAYER_CACHE.get(key)
        if payload is None and node_id:
            suffix = f":{key}"
            for candidate_key in reversed(_PLAYER_CACHE):
                if candidate_key == key or candidate_key.endswith(suffix):
                    payload = _PLAYER_CACHE[candidate_key]
                    break
        return None if payload is None else _copy_json(payload)


def _publish_player_state(node_id, payload: dict):
    data = {"node_id": None if node_id is None else str(node_id), **payload}
    _store_player_state(node_id, data)
    prompt_server = None if PromptServer is None else getattr(PromptServer, "instance", None)
    if prompt_server is not None:
        try:
            prompt_server.send_sync(PLAYER_EVENT, data, prompt_server.client_id)
        except Exception as error:  # Preview delivery must never invalidate a saved render.
            logging.warning("HR Endless Sampler finished-video player update failed: %s", error)


def _node_unique_id(node_class):
    """Read Comfy's hidden id without making direct/unit execution fail."""
    return getattr(getattr(node_class, "hidden", None), "unique_id", None)


_PROMPT_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)
if _PROMPT_SERVER is not None:
    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_video/state")
    async def hr_endless_sampler_video_state(request):
        return web.json_response(
            _player_state(request.rel_url.query.get("node_id", "")) or {},
            headers={"Cache-Control": "no-store"},
        )

    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_video/browse_output")
    async def hr_endless_sampler_video_browse_output(request):
        try:
            listing = _output_browser_listing(request.rel_url.query.get("path", ""))
        except (OSError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(listing, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_video/matching_output")
    async def hr_endless_sampler_video_matching_output(request):
        try:
            listing = _matching_output_listing(
                request.rel_url.query.get("value", ""),
                strip_counter=request.rel_url.query.get("strip_counter", "0") in {"1", "true", "yes"},
            )
        except (OSError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(listing, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_video/load_preview")
    async def hr_endless_sampler_video_load_preview(request):
        try:
            body = await request.json()
            video = str(body.get("video") or "").strip()
            node_id = str(body.get("node_id") or "").strip()
            if not video or not node_id:
                raise ValueError("video and node_id are required")
            fps = float(body.get("fps") or 0.0)
            _timeline, filename, resolved_fps, state = await asyncio.to_thread(_load_video_payload, video, fps)
            response = {"node_id": node_id, **state, "filename": filename, "fps": resolved_fps}
            # The requesting browser applies this response directly. Persist it
            # for refresh recovery without also broadcasting a duplicate event;
            # duplicate events can race when the user switches files quickly.
            _store_player_state(node_id, response)
            return web.json_response(response)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            logging.exception("HR Endless Sampler Load Video immediate preview failed")
            return web.json_response({"error": str(error)}, status=500)

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_video/upload_chunk")
    async def hr_endless_sampler_video_upload_chunk(request):
        try:
            post = await request.post()
            upload_id = str(post.get("upload_id") or "").strip()
            if not _SAFE_UPLOAD_ID.fullmatch(upload_id):
                raise ValueError("invalid upload id")
            filename = _safe_upload_filename(str(post.get("filename") or ""))
            chunk = post.get("chunk")
            if chunk is None or not getattr(chunk, "file", None):
                raise ValueError("upload chunk is missing")
            chunk_index = int(post.get("chunk_index", 0))
            total_chunks = int(post.get("total_chunks", 1))
            if total_chunks < 1 or total_chunks > 100_000 or chunk_index < 0 or chunk_index >= total_chunks:
                raise ValueError("invalid upload chunk index")

            session = Path(folder_paths.get_temp_directory()) / OUTPUT_UPLOAD_CHUNK_DIRECTORY / upload_id
            session.mkdir(parents=True, exist_ok=True)
            manifest_path = session / "manifest.json"
            manifest = {"filename": filename, "total_chunks": total_chunks}
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing != manifest:
                    raise ValueError("upload metadata changed between chunks")
            else:
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            part_path = session / f"{chunk_index:06d}.part"
            with part_path.open("wb") as output:
                while True:
                    block = chunk.file.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)

            if chunk_index + 1 < total_chunks:
                return web.json_response({"status": "partial", "chunk_index": chunk_index})

            missing = [index for index in range(total_chunks) if not (session / f"{index:06d}.part").is_file()]
            if missing:
                raise ValueError(f"upload is missing chunk {missing[0]}")
            destination = _unique_uploaded_video_path(filename)
            temporary = destination.with_suffix(destination.suffix + ".uploading")
            with temporary.open("wb") as output:
                for index in range(total_chunks):
                    with (session / f"{index:06d}.part").open("rb") as source:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            os.replace(temporary, destination)
            shutil.rmtree(session, ignore_errors=True)
            relative = destination.relative_to(Path(folder_paths.get_output_directory())).as_posix()
            logging.info("HR Endless Sampler Load Video: uploaded %s", destination)
            return web.json_response({"status": "complete", "path": relative, "name": destination.name})
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)


def _vhs_nodes():
    """Find VHS only when one of its additional formats is requested.

    Native H.264 and EXR are deliberately independent of VHS. Other formats
    stay byte-for-byte on VHS's own encoder path.
    """
    direct_error = None
    try:
        module = importlib.import_module("videohelpersuite.nodes")
        if callable(getattr(module, "get_video_formats", None)) and hasattr(module, "VideoCombine"):
            return module
    except Exception as error:
        direct_error = error

    # ComfyUI loads every custom-node directory with a path-derived package
    # name. It does not add the VHS directory itself to sys.path, so the
    # standalone import above normally fails even though VHS is active. Reuse
    # its already-loaded relative submodule instead of importing a second copy.
    for name, module in tuple(sys.modules.items()):
        if not name.endswith(".videohelpersuite.nodes"):
            continue
        if callable(getattr(module, "get_video_formats", None)) and hasattr(module, "VideoCombine"):
            return module

    raise RuntimeError(
        "The selected format needs ComfyUI-VideoHelperSuite, but its loaded videohelpersuite.nodes "
        "module could not be found. Restart ComfyUI after installing or enabling Video Helper Suite, "
        "or use video/h264-mp4 or video/exr."
    ) from direct_error


def _vhs_format_options() -> list[str]:
    try:
        formats, _widgets = _vhs_nodes().get_video_formats()
    except RuntimeError:
        formats = ["video/h264-mp4", "video/h265-mp4", "video/webm", "video/ffv1-mkv"]
    # Video Combine exposes these two animated-image choices in addition to
    # get_video_formats(). Keep the complete VHS selection, then append our
    # raw EXR-sequence mode.
    return list(dict.fromkeys(["image/gif", "image/webp", *formats, "video/exr"]))


def _vhs_pixel_formats() -> list[str]:
    values = {"auto"}
    try:
        _formats, widgets = _vhs_nodes().get_video_formats()
        for widget_list in widgets.values():
            for widget in widget_list:
                if isinstance(widget, list) and len(widget) > 1 and widget[0] == "pix_fmt" and isinstance(widget[1], list):
                    values.update(str(value) for value in widget[1])
    except RuntimeError:
        values.update({"yuv420p", "yuv420p10le", "p010le", "yuva420p", "rgba64le"})
    values.update({"half", "float"})
    return ["auto", *sorted(value for value in values if value != "auto")]


def _tensor_to_exr_bytes(image: torch.Tensor, pixel_format: str, compression: str, gamma: float = 1.0) -> bytes:
    """Encode a raw tensor as OpenEXR without normalizing or clamping it."""
    if image.ndim != 3:
        raise ValueError(f"EXR frames must be HxWxC tensors; received {tuple(image.shape)}")
    height, width, channels = (int(value) for value in image.shape)
    stream_formats = {
        1: ("grayf32le", "grayf32le"),
        3: ("gbrpf32le", "gbrpf32le"),
        4: ("gbrapf32le", "gbrapf32le"),
    }
    if channels not in stream_formats:
        raise ValueError("EXR export supports 1, 3, or 4 channels per frame")
    frame_format, stream_format = stream_formats[channels]
    data = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    if channels == 1:
        data = data[..., 0]
    codec = av.CodecContext.create("exr", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = stream_format
    codec.time_base = Fraction(1, 1)
    codec.options = {"compression": compression, "format": pixel_format, "gamma": str(float(gamma))}
    frame = av.VideoFrame.from_ndarray(data, format=frame_format)
    frame.pts = 0
    frame.time_base = codec.time_base
    return b"".join(bytes(packet) for packet in [*codec.encode(frame), *codec.encode(None)])


def _normalize_frames(images: torch.Tensor) -> torch.Tensor:
    """Accept ordinary IMAGE batches and ComfyUI video-VAE [B,T,H,W,C] output."""
    if not isinstance(images, torch.Tensor):
        raise ValueError("Save Video needs a torch IMAGE tensor")
    if images.ndim == 5:
        images = images.flatten(0, 1)
    elif images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("Save Video needs one or more HxWxC IMAGE frames")
    return images


def _inject_exr_timeline(exr_bytes: bytes, timeline: dict) -> bytes:
    try:
        from comfy_extras.nodes_images import inject_exr_metadata
        return inject_exr_metadata(exr_bytes, None, {TIMELINE_TAG: timeline}, "linear")
    except Exception as error:
        logging.warning("HR Endless Sampler could not write EXR timeline metadata: %s", error)
        return exr_bytes


def _save_exr_sequence(images: torch.Tensor, filename_prefix: str, pixel_format: str,
                       compression: str, timeline: dict, gamma: float = 1.0) -> tuple[Path, list[Path], str]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("video/exr needs one or more IMAGE frames")
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix,
        output_dir,
        int(images.shape[2]),
        int(images.shape[1]),
    )
    output_folder = Path(full_output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, image in enumerate(images):
        file_path = output_folder / f"{filename}_{counter:05}_{index:06}.exr"
        encoded = _tensor_to_exr_bytes(image, pixel_format, compression, gamma)
        # The sidecar is the durable sequence manifest. Embedding the complete
        # timeline in the first frame additionally lets a copied primary EXR
        # retain its navigation data without repeating a long prompt per frame.
        if index == 0:
            encoded = _inject_exr_timeline(encoded, timeline)
        file_path.write_bytes(encoded)
        paths.append(file_path)
    return paths[0], paths, subfolder


def _extract_vhs_files(result) -> list[Path]:
    try:
        files = result["result"][0][1]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Video Helper Suite did not return an output filename") from error
    return [Path(item) for item in files if isinstance(item, str)]


def _save_vhs_video(images: torch.Tensor, filename_prefix: str, format_name: str, fps: float,
                    pixel_format: str, crf: int, timeline: dict, *, audio=None,
                    save_output: bool = True) -> Path:
    module = _vhs_nodes()
    kwargs = {"crf": int(crf)}
    if pixel_format != "auto":
        kwargs["pix_fmt"] = pixel_format
    result = module.VideoCombine().combine_video(
        images=images,
        frame_rate=float(fps),
        loop_count=0,
        filename_prefix=filename_prefix,
        format=format_name,
        pingpong=False,
        save_output=save_output,
        extra_pnginfo={TIMELINE_TAG: timeline},
        audio=audio,
        **kwargs,
    )
    files = _extract_vhs_files(result)
    if not files:
        raise RuntimeError("Video Helper Suite completed without creating a video file")
    return files[-1]


def _save_native_h264(images: torch.Tensor, filename_prefix: str, fps: float,
                      pixel_format: str, crf: int, timeline: dict, *, audio=None,
                      save_output: bool = True) -> Path:
    """Save H.264 MP4 through ComfyUI's native VideoFromComponents API."""
    destination_root = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    full_output_folder, filename, counter, _subfolder, _resolved_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        destination_root,
        int(images.shape[2]),
        int(images.shape[1]),
    )
    output_path = Path(full_output_folder) / f"{filename}_{counter:05}_.mp4"
    use_10_bit = pixel_format in {"yuv420p10le", "p010le", "yuv444p10le", "gbrp10le"}
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(round(float(fps) * 1000), 1000),
        ),
        bit_depth=10 if use_10_bit else 8,
    )
    video.save_to(
        str(output_path),
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
        metadata={TIMELINE_TAG: timeline},
        crf=float(crf),
    )
    return output_path


def _safe_relative(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def _view_url(path: Path) -> str | None:
    for kind, root in (("output", Path(folder_paths.get_output_directory())), ("temp", Path(folder_paths.get_temp_directory())), ("input", Path(folder_paths.get_input_directory()))):
        relative = _safe_relative(path, root)
        if relative is None:
            continue
        parent = str(Path(relative).parent)
        query = {"filename": Path(relative).name, "type": kind}
        if parent not in ("", "."):
            query["subfolder"] = parent
        return "/view?" + urlencode(query)
    return None


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"'{path.name}' has no video stream")
        rate = stream.average_rate or stream.base_rate or 0
        fps = float(rate) if rate else 0.0
        frame_count = int(stream.frames or 0)
        return fps, frame_count, int(stream.width or 0), int(stream.height or 0)


def _decode_exr_sequence(paths: list[Path]) -> torch.Tensor:
    frames = []
    for path in paths:
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise ValueError(f"EXR sequence frame '{path.name}' is empty")
            if frame.format.name == "grayf32le":
                array = frame.to_ndarray(format="grayf32le")[..., None]
            elif "a" in frame.format.name:
                array = frame.to_ndarray(format="gbrapf32le")
            else:
                array = frame.to_ndarray(format="gbrpf32le")
            frames.append(torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32))
    if not frames:
        raise ValueError("EXR sequence contains no frames")
    return torch.stack(frames)


def _audio_waveform(audio) -> tuple[torch.Tensor, int]:
    """Validate ComfyUI AUDIO data and return one channels-by-samples tensor."""
    if not isinstance(audio, dict):
        raise ValueError("audio must be ComfyUI AUDIO data")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim not in (2, 3):
        raise ValueError("audio waveform must be [channels, samples] or [1, channels, samples]")
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError("Save Video supports one AUDIO batch per rendered video")
        waveform = waveform[0]
    if waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError("audio waveform is empty")
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("audio sample_rate must be a positive integer") from error
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be a positive integer")
    return waveform.detach().to(device="cpu", dtype=torch.float32).contiguous(), sample_rate


def _write_wav_audio(audio, primary_path: Path) -> Path:
    """Write an EXR sequence's float soundtrack without reducing it to PCM16."""
    waveform, sample_rate = _audio_waveform(audio)
    channels = int(waveform.shape[0])
    layouts = {1: "mono", 2: "stereo", 3: "2.1", 4: "quad", 5: "5.0", 6: "5.1", 8: "7.1"}
    layout = layouts.get(channels)
    if layout is None:
        raise ValueError(f"EXR audio sidecar does not support {channels} audio channels")
    audio_path = primary_path.with_suffix(".wav")
    output = av.open(str(audio_path), mode="w", format="wav")
    try:
        stream = output.add_stream("pcm_f32le", rate=sample_rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(
            waveform.movedim(0, 1).reshape(1, -1).numpy(),
            format="flt",
            layout=layout,
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    finally:
        output.close()
    return audio_path


def _decode_audio_file(path: Path) -> dict:
    """Decode an EXR's persistent soundtrack only for its browser proxy."""
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError(f"'{path.name}' has no audio stream")
        frames = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            channels = len(frame.layout.channels)
            if frame.format.is_planar:
                array = array.reshape(channels, frame.samples)
            else:
                array = array.reshape(frame.samples, channels).transpose(1, 0)
            frames.append(array.astype(np.float32, copy=False))
        if not frames:
            raise ValueError(f"'{path.name}' contains no audio samples")
        sample_rate = int(stream.rate or frames[0].sample_rate or 0)
    waveform = torch.from_numpy(np.ascontiguousarray(np.concatenate(frames, axis=1))).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _make_proxy(images: torch.Tensor, fps: float, timeline: dict, *, audio=None) -> Path:
    # A browser cannot display EXR directly. A temp H.264 proxy is only for the
    # node's player; the requested EXR sequence remains the untouched master.
    return _save_native_h264(
        images,
        "hr_endless_sampler_preview/finished_video",
        fps,
        "yuv420p",
        23,
        timeline,
        audio=audio,
        save_output=False,
    )


def _resolve_input_path(value: str) -> Path:
    raw = Path(str(value).strip()).expanduser()
    if raw.is_file():
        return raw.resolve()
    for root in (Path(folder_paths.get_output_directory()), Path(folder_paths.get_input_directory()), Path(folder_paths.get_temp_directory())):
        candidate = root / raw
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"video path does not exist: {value}")


def _media_from_sidecar(media_path: Path, media: dict | None) -> tuple[str, list[Path] | None]:
    if not media or media.get("kind") != "exr_sequence":
        return "video", None
    filenames = media.get("frames")
    if not isinstance(filenames, list) or not filenames:
        return "exr_sequence", [media_path]
    paths = [media_path.parent / str(filename) for filename in filenames]
    return "exr_sequence", paths


class HREndlessSamplerSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessSamplerSaveVideo",
            display_name="HR Endless Sampler Save Video",
            category="image/video",
            description=("Saves rendered frames with HR Endless Sampler chunk/shot metadata. "
                         "H.264 uses ComfyUI's native encoder; extra formats use VHS; video/exr writes an unclamped EXR sequence."),
            inputs=[
                io.Image.Input("images", optional=True, tooltip="Decoded frames to save. For unclamped H3 EXR, connect latent and vae instead."),
                io.Latent.Input("latent", optional=True, tooltip="Optional MiniMax H3 latent for direct unclamped EXR decoding."),
                io.Vae.Input("vae", optional=True, tooltip="H3 video VAE used only with latent for direct unclamped EXR decoding."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Optional decoded soundtrack. Video formats mux it; EXR writes a sibling 32-bit float WAV."),
                HREndlessTimeline.Input("timeline", optional=True, tooltip="Timeline output from HR Endless Sampler. Without it, the save is one unmarked chunk."),
                io.Float.Input("fps", default=24.0, min=0.001, max=1000.0, step=0.001,
                               tooltip="Output and player frame rate."),
                io.String.Input("filename_prefix", default="HR_Endless_Sampler"),
                io.Combo.Input("format", options=_vhs_format_options(), default="video/h264-mp4",
                               tooltip="video/h264-mp4 uses native ComfyUI. Other video formats use installed VHS definitions. video/exr writes one OpenEXR image per frame."),
                io.Combo.Input("pixel_format", options=_vhs_pixel_formats(), default="auto",
                               tooltip="For H.264 choose an 8/10-bit pixel format. Other formats use VHS options. For video/exr choose half or float."),
                io.Int.Input("crf", default=19, min=0, max=100, step=1,
                             tooltip="H.264/VHS CRF value when the selected encoder supports CRF. EXR ignores it."),
                io.Combo.Input("exr_compression", options=["none", "rle", "zip1", "zip16"], default="zip16",
                               tooltip="OpenEXR compression supported by ComfyUI's installed FFmpeg EXR encoder."),
                io.Float.Input("exr_gamma", default=1.0, min=0.001, max=100.0, step=0.001,
                               tooltip="OpenEXR encoder gamma. Keep 1.0 to preserve raw VAE tensor values."),
            ],
            outputs=[
                HREndlessTimeline.Output(display_name="timeline"),
                io.String.Output(display_name="filename"),
                io.Float.Output(display_name="fps"),
            ],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def _raw_h3_decode(cls, latent, vae) -> torch.Tensor:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("direct raw EXR decode needs the MiniMax H3 nested video/audio latent from HR Endless Sampler")
        video = samples.unbind()[0]
        stage = getattr(vae, "first_stage_model", None)
        if stage is None or not hasattr(stage, "_finalize_pixels") or not hasattr(stage, "pixel_mean") or not hasattr(stage, "pixel_std"):
            raise ValueError("direct raw EXR decode currently supports ComfyUI's MiniMax H3 video VAE")
        original_finalize = stage._finalize_pixels

        def finalize_without_clamp(part):
            return part * stage.pixel_std.to(device=part.device, dtype=torch.float32) + stage.pixel_mean.to(device=part.device, dtype=torch.float32)

        # The stock MiniMax VAE clamps inside _finalize_pixels. Temporarily
        # replace only that final conversion, preserving the actual decoder,
        # temporal tiling, and VAE device-management path.
        stage._finalize_pixels = finalize_without_clamp
        try:
            decoded = vae.decode(video)
        finally:
            stage._finalize_pixels = original_finalize
        return _normalize_frames(decoded)

    @classmethod
    def execute(cls, images=None, latent=None, vae=None, audio=None, timeline=None, fps=24.0,
                filename_prefix="HR_Endless_Sampler", format="video/h264-mp4",
                pixel_format="auto", crf=19, exr_compression="zip16", exr_gamma=1.0):
        if images is None and (latent is None or vae is None):
            raise ValueError("connect images, or connect both latent and vae for direct H3 EXR decoding")
        if format == "video/exr" and latent is not None and vae is not None:
            frames = cls._raw_h3_decode(latent, vae)
        else:
            frames = images
        frames = _normalize_frames(frames)

        resolved_fps = _number(fps, 24.0)
        if resolved_fps <= 0:
            raise ValueError("fps must be greater than zero")
        normalized_timeline = normalize_timeline(timeline, fps=resolved_fps, total_frames=int(frames.shape[0]))
        if format == "video/exr":
            exr_format = pixel_format if pixel_format in {"half", "float"} else "float"
            primary_path, frame_paths, _subfolder = _save_exr_sequence(
                frames,
                filename_prefix,
                exr_format,
                exr_compression,
                normalized_timeline,
                float(exr_gamma),
            )
            audio_path = _write_wav_audio(audio, primary_path) if audio is not None else None
            sidecar = _write_sidecar(
                primary_path,
                normalized_timeline,
                {
                    "kind": "exr_sequence",
                    "frames": [path.name for path in frame_paths],
                    **({"audio_filename": audio_path.name} if audio_path is not None else {}),
                },
            )
            proxy_path = _make_proxy(frames.clamp(0.0, 1.0), resolved_fps, normalized_timeline, audio=audio)
            media_url = _view_url(proxy_path)
            media_kind = "exr_sequence"
        elif format == "video/h264-mp4":
            primary_path = _save_native_h264(
                frames,
                filename_prefix,
                resolved_fps,
                pixel_format,
                int(crf),
                normalized_timeline,
                audio=audio,
            )
            sidecar = _write_sidecar(primary_path, normalized_timeline, {"kind": "video", "filename": primary_path.name})
            media_url = _view_url(primary_path)
            media_kind = "video"
        else:
            primary_path = _save_vhs_video(
                frames,
                filename_prefix,
                format,
                resolved_fps,
                pixel_format,
                int(crf),
                normalized_timeline,
                audio=audio,
            )
            sidecar = _write_sidecar(primary_path, normalized_timeline, {"kind": "video", "filename": primary_path.name})
            media_url = _view_url(primary_path)
            media_kind = "video"
        if media_url is None:
            raise RuntimeError(f"saved media is outside ComfyUI's viewable folders: {primary_path}")
        logging.info("HR Endless Sampler Save Video: wrote %s and timeline sidecar %s", primary_path, sidecar)
        _publish_player_state(_node_unique_id(cls), {
            "action": "media",
            "title": primary_path.name,
            "media_url": media_url,
            "media_kind": media_kind,
            "source_fps": normalized_timeline["fps"],
            "timeline": normalized_timeline,
        })
        return io.NodeOutput(normalized_timeline, str(primary_path), float(normalized_timeline["fps"]))


def _load_video_payload(video: str, fps: float = 0.0, *, decoded: dict | None = None) -> tuple[dict, str, float, dict]:
    """Probe one saved media path and build its player state.

    The browser-only endpoint leaves ``decoded`` unset and remains lightweight.
    Workflow execution supplies a dictionary, causing this same resolved media
    to be decoded once into reusable VIDEO, IMAGE, and AUDIO outputs.
    """
    input_path = _resolve_input_path(video)
    timeline, media = _read_sidecar(input_path)
    if input_path.name.endswith(SIDECAR_SUFFIX):
        if media is None:
            raise ValueError(f"timeline sidecar has no media manifest: {input_path}")
        if media.get("kind") == "video":
            filename = media.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"timeline sidecar does not name its video: {input_path}")
            input_path = input_path.parent / filename
        else:
            frame_names = media.get("frames")
            if not isinstance(frame_names, list) or not frame_names:
                raise ValueError(f"timeline sidecar does not describe an EXR sequence: {input_path}")
            input_path = input_path.parent / str(frame_names[0])
    if timeline is None:
        timeline = _read_embedded_timeline(input_path)
    media_kind, exr_paths = _media_from_sidecar(input_path, media)

    if media_kind == "exr_sequence" or input_path.suffix.lower() == ".exr":
        paths = exr_paths or [input_path]
        if not all(path.is_file() for path in paths):
            missing = next(path for path in paths if not path.is_file())
            raise ValueError(f"EXR sequence frame is missing: {missing}")
        frames = _decode_exr_sequence(paths)
        source_fps = _number((timeline or {}).get("fps"), _number(fps, 24.0) or 24.0)
        normalized_timeline = normalize_timeline(timeline, fps=source_fps, total_frames=int(frames.shape[0]))
        audio = None
        audio_filename = media.get("audio_filename") if isinstance(media, dict) else None
        if isinstance(audio_filename, str) and audio_filename:
            audio_path = input_path.parent / audio_filename
            if audio_path.is_file():
                audio = _decode_audio_file(audio_path)
            else:
                logging.warning("HR Endless Sampler Load Video: EXR soundtrack is missing: %s", audio_path)
        proxy_path = _make_proxy(frames.clamp(0.0, 1.0), normalized_timeline["fps"], normalized_timeline, audio=audio)
        media_url = _view_url(proxy_path)
        if media_url is None:
            raise RuntimeError("could not expose temporary EXR preview proxy to ComfyUI's browser")
        title = f"{input_path.name} sequence"
        filename = str(input_path)
        if decoded is not None:
            components = Types.VideoComponents(
                images=frames,
                audio=audio,
                frame_rate=Fraction(normalized_timeline["fps"]).limit_denominator(1_000_000),
            )
            decoded.update({
                "video": InputImpl.VideoFromComponents(components, bit_depth=10),
                "images": frames,
                "audio": audio,
            })
    else:
        source_fps, source_frames, _width, _height = _probe_video(input_path)
        timeline_fps = _number((timeline or {}).get("fps"), source_fps or _number(fps, 24.0) or 24.0)
        normalized_timeline = normalize_timeline(timeline, fps=timeline_fps, total_frames=source_frames)
        if decoded is not None:
            source_video = InputImpl.VideoFromFile(str(input_path))
            components = source_video.get_components()
            decoded.update({
                "video": InputImpl.VideoFromComponents(components, bit_depth=source_video.get_bit_depth()),
                "images": components.images,
                "audio": components.audio,
            })
        media_url = _view_url(input_path)
        if media_url is None:
            raise RuntimeError(
                "the video is outside ComfyUI's output, input, or temp folders. Copy it to one of those folders before loading it in the browser player."
            )
        title = input_path.name
        filename = str(input_path)
        media_kind = "video"
    state = {
        "action": "media",
        "title": title,
        "media_url": media_url,
        "media_kind": media_kind,
        "source_fps": normalized_timeline["fps"],
        "timeline": normalized_timeline,
    }
    return normalized_timeline, filename, float(normalized_timeline["fps"]), state


class HREndlessSamplerLoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessSamplerLoadVideo",
            display_name="HR Endless Sampler Load Video",
            category="image/video",
            description=("Browse ComfyUI output folders or upload a local video and open it immediately in the timeline player. "
                         "A timeline sidecar is preferred over embedded metadata."),
            inputs=[
                io.String.Input("video", default="", placeholder="output/your_render.mp4 or the first EXR frame",
                                extra_dict={"vhs_path_extensions": ["mp4", "mkv", "webm", "mov", "exr", "json"]}),
                io.Float.Input("fps", default=0.0, min=0.0, max=1000.0, step=0.001,
                               tooltip="Playback FPS. 0 uses the rate saved with the video."),
            ],
            outputs=[
                HREndlessTimeline.Output(display_name="timeline"),
                io.String.Output(display_name="filename"),
                io.Float.Output(display_name="fps"),
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Int.Output(display_name="frame_count"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
            hidden=[io.Hidden.unique_id],
            is_output_node=False,
        )

    @classmethod
    def execute(cls, video, fps=0.0):
        decoded = {}
        timeline, filename, resolved_fps, state = _load_video_payload(video, fps, decoded=decoded)
        images = decoded["images"]
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1:
            raise ValueError(f"'{filename}' did not decode into a non-empty IMAGE batch")
        frame_count = int(images.shape[0])
        height = int(images.shape[1])
        width = int(images.shape[2])
        _publish_player_state(_node_unique_id(cls), state)
        logging.info(
            "HR Endless Sampler Load Video: loaded %s (%d frames, %dx%d)",
            filename,
            frame_count,
            width,
            height,
        )
        return io.NodeOutput(
            timeline,
            filename,
            resolved_fps,
            decoded["video"],
            images,
            decoded["audio"],
            frame_count,
            width,
            height,
        )


def _repair_save_video_prompt_links(json_data):
    """Repair stale sampler widgets and output slots before prompt validation."""
    prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
    if not isinstance(prompt, dict):
        return json_data
    import nodes as comfy_nodes

    qwen_backends = {"qwen3.5", "qwen3.6", "qwen3.8"}
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or node.get("class_type") != "HREndlessSampler":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        # One development build inserted four Qwen widgets before the existing
        # tail widgets. Workflows saved with that build retain positional widget
        # values after the controls moved to the compatibility-safe tail.
        if inputs.get("director_reasoning_effort") in qwen_backends and isinstance(inputs.get("director_n_cpu_moe"), str) and "mmproj" in inputs["director_n_cpu_moe"].casefold():
            old = {name: inputs.get(name) for name in (
                "cache_gemma_preproduction", "gemma4_mtp", "pytorch_memory_fraction", "debug",
                "debug_stop_chunk", "debug_start_chunk", "director_backend", "director_model",
                "director_mmproj", "director_mtp_draft_tokens", "director_reasoning_effort",
                "director_cpu_moe", "director_n_cpu_moe",
            )}
            inputs.update(
                cache_gemma_preproduction=old["debug_stop_chunk"],
                gemma4_mtp=old["debug_start_chunk"],
                pytorch_memory_fraction=old["director_backend"],
                debug=old["director_model"],
                debug_stop_chunk=old["director_mmproj"],
                debug_start_chunk=old["director_mtp_draft_tokens"],
                director_backend=old["director_reasoning_effort"],
                director_model=old["director_cpu_moe"],
                director_mmproj=old["director_n_cpu_moe"],
                director_mtp_draft_tokens=old["cache_gemma_preproduction"],
                director_reasoning_effort=old["gemma4_mtp"],
                director_cpu_moe=old["pytorch_memory_fraction"],
                director_n_cpu_moe=old["debug"],
            )
            logging.warning("HR Endless Sampler %s repaired Qwen widget values saved by the transitional UI schema.", node_id)
        memory_fraction = inputs.get("pytorch_memory_fraction")
        if isinstance(memory_fraction, (int, float)) and not isinstance(memory_fraction, bool) and not 0.0 < float(memory_fraction) <= 1.0:
            inputs["pytorch_memory_fraction"] = 0.85
            logging.warning("HR Endless Sampler %s restored invalid PyTorch memory fraction %r to 0.85.", node_id, memory_fraction)

    expected_types = {
        "images": "IMAGE",
        "latent": "LATENT",
        "vae": "VAE",
        "audio": "AUDIO",
        "timeline": "HRENDLESS_TIMELINE",
    }

    def output_types(node_class):
        return_types = tuple(getattr(node_class, "RETURN_TYPES", ())) if node_class is not None else ()
        if return_types or node_class is None or not hasattr(node_class, "define_schema"):
            return return_types
        schema = node_class.define_schema()
        return tuple(
            getattr(getattr(type(output), "Parent", None), "io_type", None)
            for output in schema.outputs
        )
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or node.get("class_type") != "HREndlessSamplerSaveVideo":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, expected_type in expected_types.items():
            link = inputs.get(input_name)
            if not isinstance(link, list) or len(link) != 2:
                continue
            origin_id, origin_slot = link
            origin = prompt.get(str(origin_id), prompt.get(origin_id))
            origin_type = origin.get("class_type") if isinstance(origin, dict) else None
            origin_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(origin_type)
            return_types = ("LATENT", "LATENT", "STRING", "HRENDLESS_TIMELINE") if origin_type == "HREndlessSampler" else output_types(origin_class)
            if isinstance(origin_slot, int) and 0 <= origin_slot < len(return_types):
                continue
            matching_slots = [index for index, value in enumerate(return_types) if value == expected_type]
            if len(matching_slots) == 1:
                repaired_slot = matching_slots[0]
                inputs[input_name] = [origin_id, repaired_slot]
                logging.warning(
                    "HR Endless Sampler Save Video %s repaired stale %s link from %s output %r to output %d (%s).",
                    node_id, input_name, origin_id, origin_slot, repaired_slot, expected_type,
                )
            else:
                inputs.pop(input_name, None)
                logging.error(
                    "HR Endless Sampler Save Video %s removed invalid %s link from %s output %r; "
                    "runtime %s exposes %d outputs %s and has no unique %s output.",
                    node_id, input_name, origin_id, origin_slot, origin_type,
                    len(return_types), return_types, expected_type,
                )
    return json_data


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.add_on_prompt_handler(_repair_save_video_prompt_links)
