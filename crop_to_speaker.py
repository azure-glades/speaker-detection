#!/usr/bin/env ./.venv/bin/python
"""
crop_to_speaker.py
Takes an input video and produces an output video that keeps the active
speaker in a smooth 16:9 crop.

Usage
-----
python crop_to_speaker.py input.mp4 output.mp4 --token YOUR_HF_TOKEN
"""
import argparse
import av
import cv2
import numpy as np
from speaker_diarizer import SpeakerDiarizer
from face_tracker import FaceTracker
from crop_engine import CropEngine
from body_tracker import BodyTracker

# ------------------------------------------------------------------ #
def associate_faces_with_speakers(
    diar_segments,
    face_boxes_per_frame,
    fps,
    tolerance=0.5,
):
    """
    Very simple association: for each speaker segment that has exactly
    one visible face for >= 80 % of its duration, link that face to it.

    Returns dict {speaker_label : face_index}.
    face_index is the left-to-right index returned by FaceTracker.
    """
    spk_to_face = {}
    for start, end, spk in diar_segments:
        frame_start = int(start * fps)
        frame_end = int(end * fps)
        counts = {}
        for f in range(frame_start, min(frame_end, len(face_boxes_per_frame))):
            boxes = face_boxes_per_frame[f]
            if len(boxes) == 1:
                counts[0] = counts.get(0, 0) + 1
            elif len(boxes) == 2:
                counts[0] = counts.get(0, 0) + 1
                counts[1] = counts.get(1, 0) + 1

        if counts:
            best_idx = max(counts, key=counts.get)
            if counts[best_idx] / (frame_end - frame_start) >= 0.8:
                spk_to_face[spk] = best_idx
    # fallback: speaker_0 -> left face, speaker_1 -> right face
    for spk in ["SPEAKER_00", "SPEAKER_01"]:
        spk_to_face.setdefault(spk, 0 if spk.endswith("00") else 1)
    return spk_to_face
# ------------------------------------------------------------------ #
def letterbox_portrait(img):
    h, w, _ = img.shape
    target_w, target_h = 720, 1280
    scale = min(target_h / h, target_w / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return letterboxed

# ------------------------------------------------------------------ #
def draw_debug_info(img, boxes, active_spk, spk2face, frame_idx, t, segments):
    """Draw comprehensive debugging information on frame"""
    h, w = img.shape[:2]
    
    # Draw face bounding boxes with indices
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Different colors for different faces
        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for face 0, Blue for face 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'Face {i}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Display frame info
    cv2.putText(img, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'Time: {t:.2f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display active speaker info
    if active_spk:
        cv2.putText(img, f'Active: {active_spk}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if active_spk in spk2face:
            face_idx = spk2face[active_spk]
            cv2.putText(img, f'Tracking Face: {face_idx}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(img, 'No face mapping', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, 'No active speaker', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display speaker-to-face mapping
    mapping_text = f'Mapping: {spk2face}'
    cv2.putText(img, mapping_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Display current speaker segments for context
    segment_info = []
    for s, e, spk in segments:
        if abs(t - s) < 2.0 or abs(t - e) < 2.0 or (s <= t <= e):  # Show nearby segments
            status = "ACTIVE" if s <= t <= e else "NEAR"
            segment_info.append(f"{spk}: {s:.1f}-{e:.1f}s {status}")
            if len(segment_info) >= 3:  # Limit to 3 segments
                break
    
    for i, info in enumerate(segment_info):
        cv2.putText(img, info, (10, h - 50 - i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--resolution", default="720x1280", help="WxH output")
    args = parser.parse_args()

    # 0. Open containers -------------------------------------------------
    in_container = av.open(args.input)
    v_in = in_container.streams.video[0]
    a_in = in_container.streams.audio[0]
    fps = float(v_in.average_rate)

    out_w, out_h = map(int, args.resolution.split("x"))
    out_container = av.open(args.output, "w")
    v_out = out_container.add_stream("h264", rate=int(round(fps)))
    v_out.width, v_out.height = out_w, out_h
    v_out.pix_fmt = "yuv420p"
    a_out = out_container.add_stream("aac", rate=a_in.rate)

    # 1. Audio diarization (whole file) ---------------------------------
    print("Running diarization …")
    diarizer = SpeakerDiarizer(hf_token=args.token)
    audio_frames = []
    for frame in in_container.decode(audio=0):
        audio_frames.append(frame.to_ndarray().mean(axis=0))
    waveform = np.concatenate(audio_frames)
    segments = diarizer(waveform, a_in.rate)
    print("Diarization done:", segments)

    # 2. Pre-scan faces --------------------------------------------------
    print("Scanning faces …")
    in_container.seek(0)

    tracker = FaceTracker()
    # tracker = BodyTracker()
    
    face_boxes_per_frame = []
    for frame in in_container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        face_boxes_per_frame.append(tracker(img))

    # 3. Associate speakers to faces ------------------------------------
    spk2face = associate_faces_with_speakers(
        segments, face_boxes_per_frame, fps
    )
    print("Speaker -> face index map:", spk2face)

    # 4. Main processing loop ------------------------------------------
    in_container.seek(0)
    cropper = CropEngine(fps=fps)
    frame_idx = 0

    for frame in in_container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        boxes = face_boxes_per_frame[frame_idx]

        # active speaker at this frame
        t = frame_idx / fps
        active_spk = None
        for s, e, spk in segments:
            if s <= t < e:
                active_spk = spk
                break
        
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_debug_info(img.copy(), boxes, active_spk, spk2face, frame_idx, t, segments)

        if active_spk and active_spk in spk2face:
            face_idx = spk2face[active_spk]
            if face_idx < len(boxes):
                yslice, xslice = cropper.update(boxes[face_idx], img.shape)
                cropped = img[yslice, xslice]
                #cropped = cv2.resize(cropped, (out_w, out_h))
                cropped = letterbox_portrait(cropped)
            else:
                #cropped = cv2.resize(img, (out_w, out_h))
                cv2.putText(img, 'WARNING: Face not found!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cropped = letterbox_portrait(cropped)
        else:
            cv2.putText(img, 'INFO: No active speaker', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cropped = letterbox_portrait(img)

        new_frame = av.VideoFrame.from_ndarray(cropped, format="bgr24")
        for packet in v_out.encode(new_frame):
            out_container.mux(packet)

        frame_idx += 1

    # flush video packets
    for packet in v_out.encode():
        out_container.mux(packet)

    # decode and re-encode every audio frame
    for audio_frame in in_container.decode(audio=0):
        for packet in a_out.encode(audio_frame):
            out_container.mux(packet)

    # flush audio encoder
    for packet in a_out.encode():
        out_container.mux(packet)

    out_container.close()
    
    
    print("Finished →", args.output)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()