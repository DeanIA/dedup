import os
import math
import numpy as np
import av
from PIL import Image
import torch
import re
import faiss
from typing import Dict, List, Tuple, Set
import pandas as pd
from collections import defaultdict

__all__ = [
    "scan_dir",
    "create_memap",
    "process_video_directory",
    "find_duplicates",
]

def _count_segments(file_path, clip_time):
    try:
        container = av.open(file_path)
        if container.duration is None:
            return 0
        duration_seconds = container.duration / 1e6
        return int(math.ceil(duration_seconds / clip_time))
    except Exception:
        return 0

def scan_dir(video_dir, clip_time):
    total_segments = 0
    for fn in os.listdir(video_dir):
        if not fn.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        path = os.path.join(video_dir, fn)
        total_segments += _count_segments(path, clip_time=clip_time)
    return total_segments

def create_memap(file_path, dtype, shape, init_value):
    """
    Create a memory-mapped numpy array with specified properties and initialize it.

    Args:
        file_path (str): Path to the file where the memory map will be stored.
        dtype (numpy.dtype): Data type of the array elements.
        shape (tuple): Shape of the array.
        init_value (scalar): Value to initialize all elements of the array.

    Returns:
        numpy.memmap: Memory-mapped numpy array initialized with the given value.
    """
    memmap = np.lib.format.open_memmap(
    file_path,
    mode='w+', # 'w+' creates a new file or overwrites an existing one
    dtype=dtype,
    shape=shape)

    memmap[:] = init_value # Initialize arrays with values to avoid corrupt data
    return memmap 


def _extract_frame_at_current_position(container):
    """
    Extract a single frame from the container at its current seek position.
    
    Args:
        container: PyAV container object positioned at desired timestamp
        
    Returns:
        numpy.ndarray or None: Frame as RGB24 array, or None if extraction failed
    """
    frame_arr = None
    
    # CONDITION 1: Normal frame extraction
    # Try to decode the first frame at current position
    for frame in container.decode(video=0):
        frame_arr = frame.to_ndarray(format="rgb24")
        break  # Only want the first frame
    
    return frame_arr


def _create_zero_frame_fallback(container, existing_frames):
    """
    Create a black frame as fallback when frame extraction fails.
    
    Args:
        container: PyAV container object
        existing_frames: List of previously extracted frames
        
    Returns:
        numpy.ndarray: Black frame with appropriate dimensions
    """
    # CONDITION 2A: We have previous frames - use their dimensions
    if existing_frames:
        zero_frame = np.zeros_like(existing_frames[-1])
    else:
        # CONDITION 2B: No previous frames - get dimensions from video start
        # Seek to beginning and extract first frame to get video dimensions
        container.seek(0)
        for frm in container.decode(video=0):
            zero_frame = np.zeros_like(frm.to_ndarray(format="rgb24"))
            break
    
    return zero_frame


def _ensure_minimum_frames(frames_list, required_count):
    """
    Pad frame list with black frames to meet minimum count requirement.
    
    Args:
        frames_list: List of frame arrays
        required_count: Minimum number of frames needed
        
    Returns:
        list: Frame list padded to required_count length
    """
    # CONDITION 3: Insufficient frames - pad with black frames
    if len(frames_list) < required_count:
        zero_frame = np.zeros_like(frames_list[-1])
        while len(frames_list) < required_count:
            frames_list.append(zero_frame)
    
    return frames_list

def _calculate_video_segments(video_stream, clip_time):
    """
    Calculate total video duration and number of clips.
    
    Args:
        video_stream: PyAV video stream object
        clip_time: Duration of each clip in seconds
        
    Returns:
        tuple: (video_duration_seconds, num_clips)
    """
    video_duration = video_stream.duration * video_stream.time_base
    num_clips = int(math.ceil(video_duration / clip_time))
    return video_duration, num_clips


def _calculate_clip_timestamps(clip_idx, clip_time, num_samples):
    """
    Calculate frame timestamps for a specific clip.
    
    Args:
        clip_idx: Index of current clip (0-based)
        clip_time: Duration of each clip in seconds
        num_samples: Number of frames to sample per clip
        
    Returns:
        list_of_timestamps
    """
    segment_start = clip_idx * clip_time
    interval = clip_time / num_samples
    
    timestamps = []
    for frame_sample in range(num_samples):
        frame_time_stamp = segment_start + (frame_sample * interval)
        timestamps.append(frame_time_stamp)
    
    return timestamps


def _seek_to_timestamp(container, video_stream, frame_time_stamp, video_duration):
    """
    Seek container to specific timestamp with boundary checking.
    
    Args:
        container: PyAV container object
        video_stream: PyAV video stream object
        frame_time_stamp: Target timestamp in seconds
        video_duration: Total video duration in seconds
        
    Returns:
        float: Actual timestamp used (may be clamped to video_duration)
    """
    # Clamp timestamp to video duration
    if frame_time_stamp > video_duration:
        frame_time_stamp = video_duration
    
    # Convert to PTS and seek
    desired_pts = int(frame_time_stamp / video_stream.time_base)
    container.seek(desired_pts, any_frame=False, backward=True, stream=video_stream)
    return frame_time_stamp

def _sample_clips_generator(container, video_stream, clip_time):
    num_samples = 8  # xCLIP default
    if video_stream.duration is None:
        return

    video_duration, num_clips = _calculate_video_segments(video_stream, clip_time)
    for clip_idx in range(num_clips):
        timestamps = _calculate_clip_timestamps(clip_idx, clip_time, num_samples)
        frames_np = []
        for ts in timestamps:
            _seek_to_timestamp(container, video_stream, ts, video_duration)
            try:
                frame = _extract_frame_at_current_position(container)
            except av.AVError:
                frame = None

            if frame is None:
                # insert a black fallback
                frame = _create_zero_frame_fallback(container, frames_np)
            frames_np.append(frame)

        # pad/truncate to exactly num_samples
        frames_np = _ensure_minimum_frames(frames_np, num_samples)

        # yield one clip per (fn, clip_idx) **always**
        yield np.stack(frames_np)


def _convert_clips_to_pil(clip_buffer):
    """
    Convert batch of numpy video clips to PIL Image format for model processing.
    
    Args:
        clip_buffer: List of numpy arrays, each with shape (num_frames, height, width, 3)
        
    Returns:
        list: List of lists, where each inner list contains PIL Images for one clip
    """
    pil_clips = []
    for clip in clip_buffer:
        frames = [Image.fromarray(frame_np) for frame_np in clip]
        pil_clips.append(frames)
    
    return pil_clips

def _add_embeddings_to_faiss(features, id_buf, index, batch_num):
    """
    Add feature embeddings to FAISS index with proper type conversion.
    
    Args:
        features: Raw feature embeddings from model (any numpy dtype)
        id_buf: List of embedding IDs
        index: FAISS index object
        batch_num: Current batch number for logging
        
    Returns:
        tuple: (embeddings, embedding_ids) - the converted arrays that were added
    """
    embeddings = features.astype(np.float32, copy=False)
    embedding_ids = np.asarray(id_buf, dtype=np.int64)
    index.add_with_ids(embeddings, embedding_ids)
    print(f"[Batch {batch_num}] added to FAISS (index size now: {index.ntotal})")
    
    return embeddings, embedding_ids

def _store_embeddings(embeddings, embedding_ids, emb_memmap, id_memmap, write_ptr):
    """
    Store embeddings and IDs to memory-mapped files.
    
    Args:
        embeddings: Embedding feature vectors (numpy array)
        embedding_ids: Corresponding IDs for embeddings (numpy array)
        emb_memmap: Memory-mapped file for embeddings
        id_memmap: Memory-mapped file for embedding IDs
        write_ptr: Starting position to write in the memory-mapped files
        
    Returns:
        int: Updated write pointer (write_ptr + number_of_embeddings)
    """
    n = len(embedding_ids)
    emb_memmap[write_ptr : write_ptr + n, :] = embeddings
    id_memmap [write_ptr : write_ptr + n] = embedding_ids
    
    return write_ptr + n

# Initialize batch_num at the module level
batch_num = 0

def _flush_batch(clip_buf_all, id_buf_all, processor, model, write_ptr, index, emb_memmap, id_memmap):
    global batch_num
    batch_num += 1
    print(f"[Batch {batch_num}] Inference on {len(id_buf_all)} IDs: "
          f"{id_buf_all[0]} … {id_buf_all[-1]} → writing at rows "
          f"{write_ptr} … {write_ptr + len(id_buf_all) - 1}")

    # Process clips -> PIL image -> xCLIP Format torch tensor -> GPU
    pil_clips = _convert_clips_to_pil(clip_buf_all)
    inputs = processor(videos=pil_clips, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run inference 
    with torch.no_grad():
        features = model.get_video_features(**inputs).cpu().numpy()

    # Add to FAISS
    embeddings, embedding_ids = _add_embeddings_to_faiss(features, id_buf_all, index, batch_num)

    # Store embeddings and ID in .npy files 
    write_ptr = _store_embeddings(embeddings, embedding_ids, emb_memmap, id_memmap, write_ptr)
    return write_ptr


def find_duplicates(lim, distance_matrix, identity_matrix, id_array):
# Find duplicate pairs 
    pairs = []
    for query in range(len(id_array)):
        query_id = id_array[query]
        start, end = lim[query], lim[query+1]
        for i in range(start, end):
            neighbor_id = identity_matrix[i]
            distance = distance_matrix[i]
            if neighbor_id == query_id: # Skip self match
                continue
            # Only add if query_id < neighbor_id to avoid double pairs
            if query_id < neighbor_id:
                pairs.append((query_id, neighbor_id, distance))
    return pairs


def _list_video_files(video_dir):
    """
    Yield (filename, full_path) for all supported videos in sorted order.
    """
    for fn in sorted(os.listdir(video_dir)):
        if fn.lower().endswith((".mp4", ".mov", ".avi")):
            yield fn, os.path.join(video_dir, fn)

def _open_video_stream(file_path):
    """
    Open the file with PyAV and return (container, video_stream).
    """
    container = av.open(file_path)
    return container, container.streams.video[0]

def _process_clips_in_container(
    fn,
    container,
    video_stream,
    clip_time,
    batch_size,
    next_clip_id,
    name_dict,
    clip_buf_all,
    id_buf_all,
    total_clips,
    write_ptr,
    processor,
    model,
    index,
    emb_memmap,
    id_memmap
):
    """
    Drive _sample_clips_generator, assign clip-IDs, buffer clips & IDs,
    and flush whenever buffer hits batch_size.
    """
    idx = 0
    for clip_np in _sample_clips_generator(container, video_stream, clip_time):
        clip_id = next_clip_id
        name_dict[clip_id] = (fn, idx)
        next_clip_id += 1

        clip_buf_all.append(clip_np)
        id_buf_all.append(clip_id)
        total_clips += 1
        idx += 1

        if len(clip_buf_all) == batch_size:
            write_ptr = _flush_batch(
                clip_buf_all,
                id_buf_all,
                processor,
                model,
                write_ptr,
                index,
                emb_memmap,
                id_memmap
            )
            clip_buf_all.clear()
            id_buf_all.clear()

    return next_clip_id, total_clips, write_ptr


def process_video_directory(
    video_dir,
    processor,
    model,
    index,
    emb_memmap,
    batch_size,
    clip_time
) -> Tuple[List[Tuple[str,int]], int, int]:
    """
    Process all videos in `video_dir`, extract fixed‐length clips,
    generate xCLIP embeddings, index them in FAISS, and store to memmap.
    Returns:
      name_list  – list of (filename, clip_idx) in order added
      total_clips– total number of clips processed
      write_ptr  – number of vectors written/indexed
    """
    print("Start process_video_directory", flush=True)
    name_list: List[Tuple[str,int]] = []
    clip_buf: List[np.ndarray] = []
    total_clips = 0
    write_ptr = 0

    for fn, file_path in _list_video_files(video_dir):
        container, video_stream = _open_video_stream(file_path)
        idx = 0

        for clip_np in _sample_clips_generator(container, video_stream, clip_time):
            name_list.append((fn, idx))
            idx += 1
            total_clips += 1
            clip_buf.append(clip_np)

            if len(clip_buf) >= batch_size:
                batch_num = write_ptr // batch_size + 1
                n = len(clip_buf)

                print(f"    - embedding batch {batch_num}: "
                      f"adding {n} clips (total after this: {write_ptr + n})",
                      flush=True)
                
                # Convert to PIL & prepare model inputs
                pil_clips = _convert_clips_to_pil(clip_buf)
                inputs = processor(videos=pil_clips, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    features = model.get_video_features(**inputs).cpu().numpy()

                # Add to FAISS
                index.add(features.astype(np.float32, copy=False))
                
                # Store to memmap
                if emb_memmap is not None:
                    n = features.shape[0]
                    emb_memmap[write_ptr : write_ptr + n] = features
                    write_ptr += n

                    print(f"Indexed batch #{(write_ptr-1)//batch_size + 1}, "
                    f"total vectors indexed: {index.ntotal}", flush=True)

                clip_buf.clear()

        container.close()

    # flush any remaining clips
    if clip_buf:
        batch_num = write_ptr // batch_size + 1
        print(f"    ‑- embedding batch {batch_num} (final)", flush=True)

        pil_clips = _convert_clips_to_pil(clip_buf)
        inputs = processor(videos=pil_clips, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_video_features(**inputs).cpu().numpy()

        index.add(features.astype(np.float32, copy=False))
        if emb_memmap is not None:
            n = features.shape[0]
            emb_memmap[write_ptr : write_ptr + n] = features
            write_ptr += n

        clip_buf.clear()

    print(f"✅ done: {total_clips} clips added", flush=True)
    print(f"    final index.ntotal = {index.ntotal}", flush=True)
    return name_list, total_clips, write_ptr