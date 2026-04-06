import os
import asyncio
from services.ai_pipeline import AILogic
from services.vector_store import VectorStore
import imageio_ffmpeg

async def process_video_pipeline(video_path: str, video_id: str, progress_callback=lambda s, p: None):
    audio_path = f"uploads/{video_id}.mp3"
    frames_dir = f"frames/{video_id}"
    os.makedirs(frames_dir, exist_ok=True)
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # 1. Extract audio
    print(f"Extracting audio for {video_id}...")
    progress_callback("Extracting audio...", 5)
    extract_audio_cmd = [
        ffmpeg_exe, "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ]
    proc1 = await asyncio.create_subprocess_exec(
        *extract_audio_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc1.communicate()

    # 2. Extract frames (1 frame every 5 seconds)
    print(f"Extracting frames for {video_id}...")
    progress_callback("Extracting frames...", 15)
    extract_frames_cmd = [
        ffmpeg_exe, "-i", video_path, "-vf", "fps=1/5", f"{frames_dir}/frame_%04d.jpg", "-y"
    ]
    proc2 = await asyncio.create_subprocess_exec(
        *extract_frames_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc2.communicate()
    
    # 3. Transcribe audio via Groq
    print(f"Transcribing audio for {video_id}...")
    progress_callback("Transcribing audio...", 25)
    segments = await AILogic.transcribe_audio(audio_path)
    
    # 4. Caption frames via Groq
    print(f"Captioning frames for {video_id}...")
    progress_callback("Stitching frame grids...", 30)
    from PIL import Image
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    def create_grid(paths, out_path):
        images = [Image.open(p) for p in paths]
        if not images: return
        w, h = images[0].size
        grid = Image.new('RGB', (w * 2, h * 2))
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        for idx, img in enumerate(images):
            grid.paste(img, positions[idx])
        grid.save(out_path, format="JPEG", quality=70)

    batches = []
    batch_size = 4
    for i in range(0, len(frame_files), batch_size):
        batches.append(frame_files[i:i+batch_size])

    grid_tasks = []
    grid_times = []
    grid_paths = []

    for i, batch in enumerate(batches):
        batch_paths = [os.path.join(frames_dir, f) for f in batch]
        grid_path = os.path.join(frames_dir, f"grid_{i}.jpg")
        create_grid(batch_paths, grid_path)

        # Base time of the grid is the time of its first frame
        frame_time = i * batch_size * 5
        grid_times.append(frame_time)
        grid_paths.append(grid_path)
        
        # Build tasks list to run simultaneously
        grid_tasks.append(AILogic.caption_image(grid_path))

    progress_callback("Analyzing composite grids instantly...", 45)
    # Fire all AI Vision requests simultaneously, dropping 2-min wait!
    captions = await asyncio.gather(*grid_tasks)

    frame_captions = []
    for t_time, g_path, cap in zip(grid_times, grid_paths, captions):
        frame_captions.append({
            "time": t_time,
            "path": g_path,
            "caption": cap
        })
    progress_callback("Finished vision analysis", 90)

    # 5. Combine into temporally aware chunks
    print(f"Chunking data for {video_id}...")
    progress_callback("Chunking processed data...", 92)
    chunks = []
    chunk_size = 30 # Combine context every 30 seconds
    
    max_time = 0
    if segments:
        last_seg = segments[-1]
        try:
            max_time = last_seg.get('end', 0) if isinstance(last_seg, dict) else getattr(last_seg, 'end', 0)
        except Exception:
            pass
    if frame_captions:
        max_time = max(max_time, frame_captions[-1]["time"])

    current_time = 0
    while current_time <= max_time:
        end_time = current_time + chunk_size
        chunk_text = ""
        
        # Audio contextual info
        for seg in segments:
            seg_start = seg.get('start', 0) if isinstance(seg, dict) else getattr(seg, 'start', 0)
            seg_end = seg.get('end', 0) if isinstance(seg, dict) else getattr(seg, 'end', 0)
            seg_txt = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            if current_time <= seg_start < end_time or current_time < seg_end <= end_time:
                chunk_text += f"Speech [{int(seg_start)}s-{int(seg_end)}s]: {seg_txt.strip()}\n"
                
        # Visual contextual info
        chunk_frames = []
        for cap in frame_captions:
            if current_time <= cap["time"] < end_time:
                chunk_text += f"Visual at {cap['time']}s: {cap['caption'].strip()}\n"
                chunk_frames.append(cap["path"])
        
        if chunk_text.strip():
            mm_start, ss_start = divmod(int(current_time), 60)
            mm_end, ss_end = divmod(int(end_time), 60)
            ts_str = f"{mm_start:02d}:{ss_start:02d} - {mm_end:02d}:{ss_end:02d}"
            
            chunks.append({
                "text": chunk_text,
                "timestamp": ts_str,
                "frame_path": chunk_frames[0] if chunk_frames else "",
                "video_id": video_id
            })
            
        current_time += chunk_size

    # 6. Store in vector database
    print(f"Storing {len(chunks)} chunks in Vector DB...")
    progress_callback("Storing contextual embeddings...", 96)
    VectorStore.add_chunks(chunks)
    
    progress_callback("Complete", 100)
    return chunks
