import asyncio
import os
from collections.abc import Callable

import imageio_ffmpeg
from PIL import Image

from services.ai_pipeline import AILogic
from services.vector_store import VectorStore

FRAME_INTERVAL_SECONDS = 5
GRID_BATCH_SIZE = 4
MAX_VISION_CONCURRENCY = 3
SIMILARITY_HASH_SIZE = 8
SIMILARITY_DISTANCE_THRESHOLD = 6
MAX_CAPTION_REUSE_SECONDS = 120
CHUNK_SIZE_SECONDS = 30

ProgressCallback = Callable[..., None]


def average_hash(image_path: str, hash_size: int = SIMILARITY_HASH_SIZE) -> tuple[int, ...]:
    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((hash_size, hash_size))
        pixels = list(grayscale.getdata())
    mean_value = sum(pixels) / len(pixels)
    return tuple(1 if pixel >= mean_value else 0 for pixel in pixels)


def hamming_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(1 for left_bit, right_bit in zip(left, right) if left_bit != right_bit)


async def run_ffmpeg_command(*command: str) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout, stderr


def build_grid_image(paths: list[str], out_path: str):
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return

    try:
        width, height = images[0].size
        grid = Image.new("RGB", (width * 2, height * 2))
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        for index, image in enumerate(images):
            grid.paste(image, positions[index])
        grid.save(out_path, format="JPEG", quality=70)
    finally:
        for image in images:
            image.close()


def build_chunks(
    *,
    video_id: str,
    segments,
    frame_captions: list[dict],
    stage: str,
):
    chunks = []
    max_time = 0

    if segments:
        last_segment = segments[-1]
        try:
            max_time = last_segment.get("end", 0) if isinstance(last_segment, dict) else getattr(last_segment, "end", 0)
        except Exception:
            pass

    if frame_captions:
        max_time = max(max_time, frame_captions[-1]["time"])

    current_time = 0
    while current_time <= max_time:
        end_time = current_time + CHUNK_SIZE_SECONDS
        chunk_text = ""
        latest_visual_caption = None
        latest_visual_time = None

        for segment in segments:
            segment_start = segment.get("start", 0) if isinstance(segment, dict) else getattr(segment, "start", 0)
            segment_end = segment.get("end", 0) if isinstance(segment, dict) else getattr(segment, "end", 0)
            segment_text = segment.get("text", "") if isinstance(segment, dict) else getattr(segment, "text", "")
            if current_time <= segment_start < end_time or current_time < segment_end <= end_time:
                chunk_text += f"Speech [{int(segment_start)}s-{int(segment_end)}s]: {segment_text.strip()}\n"

        chunk_frames = []
        for caption in frame_captions:
            if caption["time"] <= end_time:
                latest_visual_caption = caption["caption"]
                latest_visual_time = caption["time"]
            if current_time <= caption["time"] < end_time:
                chunk_text += f"Visual at {caption['time']}s: {caption['caption'].strip()}\n"
                chunk_frames.append(caption["path"])

        if not chunk_frames and latest_visual_caption is not None and latest_visual_time is not None:
            chunk_text += (
                f"Visual context continuing from {latest_visual_time}s: "
                f"{latest_visual_caption.strip()}\n"
            )

        if chunk_text.strip():
            start_minutes, start_seconds = divmod(int(current_time), 60)
            end_minutes, end_seconds = divmod(int(end_time), 60)
            timestamp = f"{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}"
            chunks.append(
                {
                    "text": chunk_text,
                    "timestamp": timestamp,
                    "frame_path": chunk_frames[0] if chunk_frames else "",
                    "video_id": video_id,
                    "stage": stage,
                }
            )

        current_time += CHUNK_SIZE_SECONDS

    return chunks


async def extract_media(video_path: str, audio_path: str, frames_dir: str):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    extract_audio_cmd = [
        ffmpeg_exe, "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ]
    extract_frames_cmd = [
        ffmpeg_exe,
        "-i",
        video_path,
        "-vf",
        f"fps=1/{FRAME_INTERVAL_SECONDS}",
        f"{frames_dir}/frame_%04d.jpg",
        "-y",
    ]

    audio_task = asyncio.create_task(run_ffmpeg_command(*extract_audio_cmd))
    frames_task = asyncio.create_task(run_ffmpeg_command(*extract_frames_cmd))

    audio_returncode, _, audio_stderr = await audio_task
    if audio_returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {audio_stderr.decode('utf-8', errors='ignore')[:500]}")

    frames_returncode, _, frames_stderr = await frames_task
    if frames_returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {frames_stderr.decode('utf-8', errors='ignore')[:500]}")


async def caption_visual_changes(frames_dir: str, progress_callback: ProgressCallback) -> list[dict]:
    frame_files = sorted(file_name for file_name in os.listdir(frames_dir) if file_name.endswith(".jpg"))
    if not frame_files:
        return []

    batches = []
    for index in range(0, len(frame_files), GRID_BATCH_SIZE):
        batches.append(frame_files[index:index + GRID_BATCH_SIZE])

    grid_entries = []
    for index, batch in enumerate(batches):
        batch_paths = [os.path.join(frames_dir, file_name) for file_name in batch]
        grid_path = os.path.join(frames_dir, f"grid_{index}.jpg")
        build_grid_image(batch_paths, grid_path)
        grid_entries.append(
            {
                "time": index * GRID_BATCH_SIZE * FRAME_INTERVAL_SECONDS,
                "path": grid_path,
                "hash": average_hash(grid_path),
                "caption": "",
            }
        )

    caption_jobs: list[tuple[int, str]] = []
    last_captioned_index: int | None = None
    for index, entry in enumerate(grid_entries):
        if last_captioned_index is None:
            caption_jobs.append((index, entry["path"]))
            last_captioned_index = index
            continue

        previous_entry = grid_entries[last_captioned_index]
        distance = hamming_distance(entry["hash"], previous_entry["hash"])
        time_gap = entry["time"] - previous_entry["time"]

        if distance <= SIMILARITY_DISTANCE_THRESHOLD and time_gap <= MAX_CAPTION_REUSE_SECONDS:
            entry["caption"] = previous_entry["caption"]
        else:
            caption_jobs.append((index, entry["path"]))
            last_captioned_index = index

    async def caption_with_limit(grid_path: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            return await AILogic.caption_image(grid_path)

    progress_callback("Analyzing distinct visual changes...", 78, phase="visual_enrichment")
    semaphore = asyncio.Semaphore(MAX_VISION_CONCURRENCY)
    captions = await asyncio.gather(
        *(caption_with_limit(grid_path, semaphore) for _, grid_path in caption_jobs)
    )

    for (index, _), caption in zip(caption_jobs, captions):
        grid_entries[index]["caption"] = caption

    last_caption = "Vision caption unavailable for this moment."
    for entry in grid_entries:
        if entry["caption"]:
            last_caption = entry["caption"]
        else:
            entry["caption"] = last_caption

    return [
        {
            "time": entry["time"],
            "path": entry["path"],
            "caption": entry["caption"],
        }
        for entry in grid_entries
    ]


async def enrich_visual_index(
    *,
    video_id: str,
    frames_dir: str,
    segments,
    progress_callback: ProgressCallback,
):
    try:
        frame_captions = await caption_visual_changes(frames_dir, progress_callback)
        progress_callback("Rebuilding multimodal chunks...", 92, phase="reindexing")
        enriched_chunks = build_chunks(
            video_id=video_id,
            segments=segments,
            frame_captions=frame_captions,
            stage="enriched",
        )
        VectorStore.replace_video_chunks(video_id, enriched_chunks)
        progress_callback(
            "Complete",
            100,
            phase="complete",
            is_search_ready=True,
            is_complete=True,
            search_quality="enriched",
        )
    except Exception as error:
        print(f"Visual enrichment error: {error}")
        progress_callback(
            "Search ready. Visual refinement hit a recoverable error.",
            100,
            phase="degraded_complete",
            is_search_ready=True,
            is_complete=True,
            search_quality="transcript_only",
            warning="Visual refinement was skipped. Transcript search still works.",
        )


async def process_video_pipeline(video_path: str, video_id: str, progress_callback: ProgressCallback = lambda *args, **kwargs: None):
    audio_path = f"uploads/{video_id}.mp3"
    frames_dir = f"frames/{video_id}"
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Extracting media for {video_id}...")
    progress_callback("Extracting audio and frames...", 10, phase="extracting")
    await extract_media(video_path, audio_path, frames_dir)

    print(f"Transcribing audio for {video_id}...")
    progress_callback("Transcribing audio for fast search...", 35, phase="transcribing")
    segments = await AILogic.transcribe_audio(audio_path)

    progress_callback("Building transcript-first search index...", 55, phase="initial_index")
    transcript_chunks = build_chunks(
        video_id=video_id,
        segments=segments,
        frame_captions=[],
        stage="transcript_only",
    )

    if transcript_chunks:
        VectorStore.replace_video_chunks(video_id, transcript_chunks)

        progress_callback(
            "Search ready. Refining visual context in background...",
            68,
            phase="search_ready",
            is_search_ready=True,
            is_complete=False,
            search_quality="transcript_only",
        )

        asyncio.create_task(
            enrich_visual_index(
                video_id=video_id,
                frames_dir=frames_dir,
                segments=segments,
                progress_callback=progress_callback,
            )
        )

        return {
            "chunks": transcript_chunks,
            "search_ready": True,
            "processing_complete": False,
            "search_quality": "transcript_only",
        }

    progress_callback(
        "No speech detected. Building visual-first index...",
        68,
        phase="visual_bootstrap",
        is_search_ready=False,
        is_complete=False,
        search_quality="pending",
    )

    frame_captions = await caption_visual_changes(frames_dir, progress_callback)
    enriched_chunks = build_chunks(
        video_id=video_id,
        segments=segments,
        frame_captions=frame_captions,
        stage="enriched",
    )
    VectorStore.replace_video_chunks(video_id, enriched_chunks)
    progress_callback(
        "Complete",
        100,
        phase="complete",
        is_search_ready=True,
        is_complete=True,
        search_quality="enriched",
    )
    return {
        "chunks": enriched_chunks,
        "search_ready": True,
        "processing_complete": True,
        "search_quality": "enriched",
    }
