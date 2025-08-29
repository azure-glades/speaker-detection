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
        
        # Count which face index is most consistently present
        face_presence = {}
        total_frames = 0
        
        for f in range(frame_start, min(frame_end, len(face_boxes_per_frame))):
            boxes = face_boxes_per_frame[f]
            if len(boxes) >= 1:  # At least one face visible
                total_frames += 1
                # Count which face positions are occupied
                for i in range(len(boxes)):
                    face_presence[i] = face_presence.get(i, 0) + 1
        
        if total_frames > 0 and face_presence:
            # Find the face index that appears most consistently
            best_face_idx = max(face_presence, key=face_presence.get)
            overlap_ratio = face_presence[best_face_idx] / total_frames
            
            if overlap_ratio >= min_overlap_ratio:
                spk_to_face[spk] = best_face_idx
    
    # Fallback: speaker_0 -> left face (index 0), speaker_1 -> right face (index 1)
    for spk in ["SPEAKER_00", "SPEAKER_01"]:
        if spk not in spk_to_face:
            spk_to_face[spk] = 0 if spk.endswith("00") else 1
            
    return spk_to_face


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--resolution", default="1280x720", help="WxH output")
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

        if active_spk and active_spk in spk2face:
            face_idx = spk2face[active_spk]
            if face_idx < len(boxes):
                yslice, xslice = cropper.update(boxes[face_idx], img.shape)
                cropped = img[yslice, xslice]
                cropped = cv2.resize(cropped, (out_w, out_h))
            else:
                cropped = cv2.resize(img, (out_w, out_h))
        else:
            cropped = cv2.resize(img, (out_w, out_h))

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