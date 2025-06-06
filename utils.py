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
    # Directory and file utilities
    'scan_dir',
    'create_memap',
    
    # Video processing functions
    'process_video_directory',
    
    # Duplicate detection functions
    'find_video_duplicates',
    'reconstruct_filename',
    'prepare_search_matrices',
    
    # Multi-threshold analysis functions 
    'find_duplicates_at_threshold',
    'create_duplicate_mapping',
    'analyze_files_for_threshold',        
    'analyze_multiple_thresholds',    
    'create_multi_threshold_analysis',      
    
    # Private functions
    '_count_segments',
    '_sample_clips_generator',
    '_flush_batch',
    '_extract_frame_at_current_position',
    '_create_zero_frame_fallback',
    '_ensure_minimum_frames',
    '_calculate_video_segments',
    '_calculate_clip_timestamps',
    '_seek_to_timestamp',
    '_convert_clips_to_pil',
    '_add_embeddings_to_faiss',
    '_store_embeddings',
    '_load_embeddings_and_index',
    '_create_id_to_row_mapping',
    '_convert_index_matrix_to_row_indices',
    '_find_duplicate_pairs'
]


def _count_segments(file_path, clip_time=10):
    try:
        container = av.open(file_path)
        if container.duration is None:
            return 0
        duration_seconds = container.duration / 1e6
        return int(math.ceil(duration_seconds / clip_time))
    except Exception:
        return 0

def scan_dir(video_dir):
    total_segments = 0
    for fn in os.listdir(video_dir):
        if not fn.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        path = os.path.join(video_dir, fn)
        total_segments += _count_segments(path, clip_time=10)
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
    
    # Calculate video segments
    video_duration, num_clips = _calculate_video_segments(video_stream, clip_time)
    
    for clip_idx in range(num_clips):
        # Calculate timestamps for clip
        timestamps = _calculate_clip_timestamps(clip_idx, clip_time, num_samples)
        frames_np = []
        for frame_time_stamp in timestamps:
            # Seek to timestamp
            actual_timestamp = _seek_to_timestamp(container, video_stream, frame_time_stamp, video_duration)
            # Extract frame 
            frame_arr = _extract_frame_at_current_position(container)
            if frame_arr is None: # Incase extraction failed 
                zero_frame = _create_zero_frame_fallback(container, frames_np)
                frames_np.append(zero_frame)
            else:
                frames_np.append(frame_arr)
        frames_np = _ensure_minimum_frames(frames_np, num_samples)
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


def process_video_directory(video_dir, processor, model, index, emb_memmap, id_memmap, batch_size):
    """
    Process all video files in a directory, extract clips, generate embeddings, and index them.
    
    This function processes videos in batches to manage memory efficiently. It extracts clips
    from each video, generates embeddings using a pre-trained model, and stores them in 
    memory-mapped files and a FAISS index for similarity search.
    
    Args:
        video_dir (str): Path to directory containing video files (.mp4, .mov, .avi)
        processor: Video processor for preparing clips for the model (e.g., XClip processor)
        model: Pre-trained model for generating video embeddings (e.g., XClip model)
        index (faiss.Index): FAISS index for storing and searching embeddings
        emb_memmap (numpy.memmap): Memory-mapped array for storing embeddings
        id_memmap (numpy.memmap): Memory-mapped array for storing clip IDs
        batch_size (int, optional): Number of clips to process per batch.
    
    Returns:
        tuple: (total_clips_processed, final_write_pointer)
            - total_clips_processed (int): Total number of video clips processed
            - final_write_pointer (int): Final position in memory-mapped arrays
    
    Raises:
        FileNotFoundError: If video_dir doesn't exist
        av.error.InvalidDataError: If video files are corrupted
    
    Notes:
        - Only processes video files with extensions: .mp4, .mov, .avi
        - Expects filenames with pattern containing "_(\d+)" for clip numbering
        - Uses 10-second clips with 8 frames per clip (hardcoded in _sample_clips_generator)
        - Memory usage scales with batch_size (typical: ~2.5GB for batch_size=256)
    """
    # Initialize batch buffers and counters
    clip_buf_all = []    # Buffer for video clip arrays
    id_buf_all = []      # Buffer for corresponding clip IDs
    total_clips = 0      # Counter for total clips processed
    write_ptr = 0        # Pointer for writing to memory-mapped arrays
    
    # Process each video file in the directory (sorted for reproducibility)
    for fn in sorted(os.listdir(video_dir)):
        # Skip non-video files
        if not fn.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        # Open video container and get video stream
        file_path = os.path.join(video_dir, fn)
        container = av.open(file_path)
        video_stream = container.streams.video[0]

        # Extract clip number from filename (e.g., "video_1234.mp4" → "1234")
        match = re.search(r"_(\d+)", fn)
        clip_num_prefix = match.group(1) if match else "0000"
        idx = 0  # Index for clips within this video

        # Generate clips from the video and process them
        for clip_np in _sample_clips_generator(container, video_stream, clip_time=10):
            # Create unique clip ID: concatenate file number + clip index
            clip_id = int(f"{clip_num_prefix}{idx}")
            
            # Add clip and ID to batch buffers
            clip_buf_all.append(clip_np)
            id_buf_all.append(clip_id)
            total_clips += 1
            idx += 1

            # Process batch when it reaches the specified size
            if len(clip_buf_all) == batch_size:
                write_ptr = _flush_batch(clip_buf_all, 
                                        id_buf_all, processor, 
                                        model, write_ptr, 
                                        index, emb_memmap, id_memmap)
                # Clear buffers for next batch
                clip_buf_all, id_buf_all = [], []

        # Clean up video container
        container.close()

    # Process any remaining clips in the final batch
    if clip_buf_all:
        write_ptr = _flush_batch(
            clip_buf_all, id_buf_all,
            processor, model, write_ptr, 
            index, emb_memmap, id_memmap)

    # Print processing summary
    print(f"Total segments written: {total_clips}")
    print(f"FAISS index size: {index.ntotal}")
    
    return total_clips, write_ptr



def _load_embeddings_and_index(emb_file: str, index_path: str) -> Tuple[np.ndarray, faiss.Index]:
    """
    Load embeddings matrix and FAISS index from files.
    
    Args:
        emb_file: Path to embeddings numpy file
        index_path: Path to FAISS index file
    
    Returns:
        Tuple of (embeddings matrix, FAISS index)
    """
    emb_matrix = np.load(emb_file)
    index = faiss.read_index(index_path)
    return emb_matrix, index

def _create_id_to_row_mapping(faiss_ids: np.ndarray) -> Dict[int, int]:
    """
    Create mapping from FAISS IDs to row indices.
    
    Args:
        faiss_ids: Array of FAISS IDs
    
    Returns:
        Dictionary mapping ID to row index
    """
    return {int(uid): i for i, uid in enumerate(faiss_ids)}

def _convert_index_matrix_to_row_indices(
    index_matrix: np.ndarray, 
    id_to_row: Dict[int, int]
) -> np.ndarray:
    """
    Convert FAISS index matrix from IDs to row indices.
    
    Args:
        index_matrix: Matrix of neighbor IDs from FAISS search
        id_to_row: Mapping from ID to row index
    
    Returns:
        Matrix with IDs converted to row indices (-1 for not found)
    """
    num_embeddings, num_neigh = index_matrix.shape
    id_rows = np.full_like(index_matrix, fill_value=-1, dtype=np.int64)
    
    for r in range(num_embeddings):
        for c in range(num_neigh):
            raw_id = int(index_matrix[r, c])
            row_idx = id_to_row.get(raw_id, None)
            if row_idx is not None:
                id_rows[r, c] = row_idx
            else:
                id_rows[r, c] = -1  # ID wasn't found
    
    return id_rows

def _find_duplicate_pairs(
    distance_matrix: np.ndarray,
    id_rows: np.ndarray,
    faiss_ids: np.ndarray,
    similarity_threshold: float = 0.3
) -> List[Tuple[int, int]]:
    """
    Find duplicate pairs based on similarity threshold.
    
    Args:
        distance_matrix: Matrix of distances from FAISS search
        id_rows: Matrix of row indices for neighbors
        faiss_ids: Array of FAISS IDs
        similarity_threshold: Distance threshold for considering duplicates
    
    Returns:
        List of duplicate pairs (id1, id2)
    """
    duplicates_ids = []
    num_embeddings = distance_matrix.shape[0]
    
    for r in range(num_embeddings):
        src_id = int(faiss_ids[r])
        # Skip c=0 (self)
        dists = distance_matrix[r, 1:]
        neigh_rows = id_rows[r, 1:]
        
        # Find positions of duplicates
        dup_positions = []
        for idx_offset, dist in enumerate(dists):
            if dist < similarity_threshold and neigh_rows[idx_offset] >= 0:
                dup_positions.append(idx_offset)
        
        # Add duplicate pairs (avoid double-counting)
        for pos in dup_positions:
            neigh_row = int(neigh_rows[pos])
            if r < neigh_row:  # avoid double-counting pairs
                neigh_id = int(faiss_ids[neigh_row])
                duplicates_ids.append((src_id, neigh_id))
    
    return duplicates_ids

def find_video_duplicates(
    emb_file: str, 
    index_path: str, 
    similarity_threshold: float = 0.3
) -> List[Tuple[int, int]]:
    """
    Main function to find duplicate video clips.
    
    Args:
        emb_file: Path to embeddings file
        index_path: Path to FAISS index file
        similarity_threshold: Distance threshold for duplicates
    
    Returns:
        List of duplicate pairs (id1, id2)
    """
    # Load data
    emb_matrix, index = _load_embeddings_and_index(emb_file, index_path)
    
    # Perform search
    num_embeddings = emb_matrix.shape[0]
    distance_matrix, index_matrix = index.search(emb_matrix, num_embeddings)
    
    # Get FAISS IDs and create mappings
    faiss_ids = faiss.vector_to_array(index.id_map)
    id_to_row = _create_id_to_row_mapping(faiss_ids)
    
    print(f"num_embeddings: {num_embeddings}, faiss_ids size: {faiss_ids.shape[0]}")
    
    # Convert index matrix to row indices
    id_rows = _convert_index_matrix_to_row_indices(index_matrix, id_to_row)
    
    # Find duplicates
    duplicates_ids = _find_duplicate_pairs(
        distance_matrix, id_rows, faiss_ids, similarity_threshold
    )
    
    return duplicates_ids


def reconstruct_filename(clip_id):
    """Convert clip ID to filename format"""
    clip_id_str = str(clip_id)
    clip_num = clip_id_str[:4].zfill(4)
    segment_idx = clip_id_str[4:] if len(clip_id_str) > 4 else "0"
    filename = f"TNS_{clip_num}_.mp4"
    return filename, segment_idx

def find_duplicates_at_threshold(distance_matrix, id_rows, faiss_ids, threshold):
    """Find duplicates for a specific threshold without recomputing search"""
    duplicates_ids = []
    num_embeddings = len(faiss_ids)  
    
    for r in range(num_embeddings):
        src_id = int(faiss_ids[r])
        # Skip c=0 (self)
        dists = distance_matrix[r, 1:]
        neigh_rows = id_rows[r, 1:]
        
        # Find positions of duplicates
        dup_positions = []
        for idx_offset, dist in enumerate(dists):
            # FIXED: Use <= instead of < to include threshold value
            # Also exclude self-matches for non-zero thresholds
            if threshold == 0.0:
                # For threshold 0.0, only exact matches (but not self-matches)
                if dist == 0.0 and neigh_rows[idx_offset] >= 0:
                    dup_positions.append(idx_offset)
            else:
                # For other thresholds, include distances <= threshold but exclude self-matches
                if dist <= threshold and dist > 0.0 and neigh_rows[idx_offset] >= 0:
                    dup_positions.append(idx_offset)
        
        # Add duplicate pairs (avoid double-counting)
        for pos in dup_positions:
            neigh_row = int(neigh_rows[pos])
            # Bounds check
            if neigh_row >= len(faiss_ids):
                print(f"Warning: neigh_row {neigh_row} exceeds bounds [0, {len(faiss_ids)-1}]")
                continue
                
            if r < neigh_row:  # avoid double-counting pairs
                neigh_id = int(faiss_ids[neigh_row])
                duplicates_ids.append((src_id, neigh_id))
    
    return duplicates_ids

def create_duplicate_mapping(duplicates_ids):
    """Create bidirectional mapping of duplicates"""
    duplicate_mapping = defaultdict(list)
    duplicate_ids = set()
    
    for id1, id2 in duplicates_ids:
        duplicate_mapping[id1].append(id2)
        duplicate_mapping[id2].append(id1)
        duplicate_ids.update([id1, id2])
    
    return duplicate_mapping, duplicate_ids

def analyze_files_for_threshold(all_clip_ids, duplicate_mapping, duplicate_ids):
    """Analyze each file's duplicate status"""
    file_stats = defaultdict(lambda: {
        'video_length_clips': 0,
        'duplicate_names': set(),
        'is_duplicate': False
    })
    
    for clip_id in all_clip_ids:
        filename = reconstruct_filename(clip_id)[0]
        file_stats[filename]['video_length_clips'] += 1
        
        if clip_id in duplicate_ids:
            file_stats[filename]['is_duplicate'] = True
            # Add filenames of videos this clip is duplicated with
            for other_id in duplicate_mapping.get(clip_id, []):
                other_filename = reconstruct_filename(other_id)[0]
                if other_filename != filename:  # Don't include self
                    file_stats[filename]['duplicate_names'].add(other_filename)
    
    return file_stats

def analyze_multiple_thresholds(distance_matrix, id_rows, faiss_ids, thresholds):
    """
    Analyze duplicates across multiple thresholds and create pivot-style results.
    
    Args:
        distance_matrix: Matrix of distances from FAISS search
        id_rows: Matrix of row indices for neighbors  
        faiss_ids: Array of FAISS IDs
        thresholds: List of threshold values to analyze
        
    Returns:
        pd.DataFrame: Results with threshold columns side by side
    """
    all_clip_ids = set(faiss_ids)
    
    # Initialize results structure
    file_results = defaultdict(lambda: {
        'File_name': '',
        'Video_length': 0
    })
    
    # Process each threshold
    for threshold in thresholds:
        print(f"Processing threshold: {threshold:.4f}")  # Changed from .3f to .4f
        
        # Find duplicates for this threshold
        duplicates_ids = find_duplicates_at_threshold(distance_matrix, id_rows, faiss_ids, threshold)
        duplicate_mapping, duplicate_ids = create_duplicate_mapping(duplicates_ids)
        file_stats = analyze_files_for_threshold(all_clip_ids, duplicate_mapping, duplicate_ids)
        
        # Count files that have no duplicates
        files_with_no_duplicates = [f for f, s in file_stats.items() if not s['is_duplicate']]
        non_duplicate_file_count = len(files_with_no_duplicates)
        
        # Update results for each file
        for filename, stats in file_stats.items():
            if file_results[filename]['File_name'] == '':
                file_results[filename]['File_name'] = filename
                file_results[filename]['Video_length'] = stats['video_length_clips']
            
            duplicate_names_str = ','.join(sorted(stats['duplicate_names'])) if stats['duplicate_names'] else 'None'
            file_results[filename][f'Threshold_{threshold:.4f}'] = threshold  # Changed to .4f
            file_results[filename][f'Duplicate_names_{threshold:.4f}'] = duplicate_names_str  # Changed to .4f
            file_results[filename][f'Is_duplicate_{threshold:.4f}'] = stats['is_duplicate']  # Changed to .4f
            file_results[filename][f'Total_non_duplicates_{threshold:.4f}'] = non_duplicate_file_count  # Changed to .4f
    
    # Convert to DataFrame
    results_list = list(file_results.values())
    df = pd.DataFrame(results_list)
    
    # Reorder columns to group threshold-related columns together
    base_cols = ['File_name', 'Video_length']
    threshold_cols = []
    
    for threshold in sorted(thresholds):
        threshold_cols.extend([
            f'Threshold_{threshold:.4f}',  # Changed to .4f
            f'Duplicate_names_{threshold:.4f}',  # Changed to .4f
            f'Is_duplicate_{threshold:.4f}',  # Changed to .4f
            f'Total_non_duplicates_{threshold:.4f}'  # Changed to .4f
        ])
    
    # Reorder DataFrame columns
    df = df[base_cols + threshold_cols]
    
    return df

def create_multi_threshold_analysis(index, emb_file, thresholds):
    """
    Complete pipeline for multi-threshold duplicate analysis with pivot results.
    
    Args:
        index: FAISS index object
        emb_file: Path to embeddings numpy file
        thresholds: List of threshold values to analyze
        
    Returns:
        pd.DataFrame: Results with all thresholds in adjacent columns
    """
    # Prepare search matrices (only once)
    distance_matrix, id_rows, faiss_ids, all_clip_ids = prepare_search_matrices(index, emb_file)
    
    # Analyze all thresholds with pivot structure
    results_df = analyze_multiple_thresholds(distance_matrix, id_rows, faiss_ids, thresholds)
    
    return results_df, distance_matrix


def prepare_search_matrices(index, emb_file):
    """
    Prepare distance and index matrices for multi-threshold duplicate detection.
    
    Args:
        index: FAISS index object
        emb_file: Path to embeddings numpy file
        
    Returns:
        tuple: (distance_matrix, id_rows, faiss_ids, all_clip_ids)
    """
    print("Loading embeddings and performing FAISS search...")
    
    # Load embeddings and perform search
    emb_matrix = np.load(emb_file)
    num_embeddings = emb_matrix.shape[0]
    
    # FIX: Don't search for ALL embeddings as neighbors - use a reasonable k value
    k = min(100, num_embeddings)  # Search for top 100 neighbors (or less if fewer embeddings)
    distance_matrix, index_matrix = index.search(emb_matrix, k)
    
    # Get FAISS IDs
    faiss_ids = faiss.vector_to_array(index.id_map)
    all_clip_ids = set(faiss_ids)
    
    # Create ID to row mapping
    id_to_row = {int(uid): i for i, uid in enumerate(faiss_ids)}
    
    # Convert index matrix to row indices
    id_rows = np.full_like(index_matrix, fill_value=-1, dtype=np.int64)
    for r in range(num_embeddings):
        for c in range(index_matrix.shape[1]):
            raw_id = int(index_matrix[r, c])
            row_idx = id_to_row.get(raw_id, None)
            if row_idx is not None:
                id_rows[r, c] = row_idx
    
    print(f"Prepared matrices for {num_embeddings} embeddings with k={k} neighbors")
    return distance_matrix, id_rows, faiss_ids, all_clip_ids