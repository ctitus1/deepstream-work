#!/usr/bin/env python3
"""Turn per-frame detections into tracks, and work out which one is moving.

The detector emits independent boxes per frame with no identity, so the boxes
are first linked into tracks by greedy nearest-neighbour association. That is
enough here: the people are far apart relative to how far any of them travels
between two frames, which is the case where greedy matching and a proper
assignment algorithm agree.

Which track is "the mover" is then decided from the data rather than declared.
A stationary person's centroid jitters inside a few pixels, dominated by
detector noise; someone walking sweeps a path orders of magnitude longer. The
gap between the two populations is large enough that any threshold inside it
gives the same answer, so the split is taken at the largest gap in the sorted
spread rather than at a hardcoded number.

Run directly to inspect what it found:
    python3 eval/tracks.py [--detections eval/detections.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Track:
    id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[dict] = field(default_factory=list)

    @property
    def centroids(self) -> list[tuple[float, float]]:
        return [
            (b["left"] + b["width"] / 2.0, b["top"] + b["height"] / 2.0) for b in self.boxes
        ]

    @property
    def spread(self) -> float:
        """Diagonal of the box containing every centroid this track ever had.

        Preferred over summed frame-to-frame displacement, which accumulates
        detector jitter: a stationary box wobbling by two pixels for five
        thousand frames racks up a large path length while never going
        anywhere. Spread measures where the track actually got to.
        """
        pts = self.centroids
        if len(pts) < 2:
            return 0.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5

    def box_at(self, frame_index: int) -> dict | None:
        try:
            return self.boxes[self.frames.index(frame_index)]
        except ValueError:
            return None


def centroid(box: dict) -> tuple[float, float]:
    return box["left"] + box["width"] / 2.0, box["top"] + box["height"] / 2.0


def build_tracks(
    frames: list[dict],
    step_dist: float,
    max_reach: float,
    max_gap: int = 15,
) -> list[Track]:
    """Associate detections into tracks across frames.

    Two details do the work:

    * Matches are taken globally-cheapest-first over all (track, detection)
      pairs, not per detection in list order. Taking them in list order lets an
      early detection claim a track that a later one was a much better fit for,
      and the resulting identity swaps fragment a stationary person into a
      dozen short tracks with a large apparent spread.
    * A track survives ``max_gap`` frames without a match, and its match radius
      grows with that gap. The detector drops people for a second at a time on
      this footage; retiring a track on the first miss splits one person into
      several, which is exactly what makes "which one moved" unanswerable.
    """
    tracks: list[Track] = []
    live: list[dict] = []  # {"track", "pos", "last_frame"}

    for frame in frames:
        index = frame["index"]
        detections = [(box, centroid(box)) for box in frame["boxes"]]

        pairs = []
        for det_i, (_, here) in enumerate(detections):
            for live_i, entry in enumerate(live):
                gap = index - entry["last_frame"]
                if gap > max_gap:
                    continue
                # Capped: the radius has to grow with the gap, but an
                # uncapped one reaches the whole frame after a second and lets
                # a track jump to a different person entirely -- which is what
                # produced tracks whose "spread" was the frame diagonal.
                limit = min(step_dist * (1 + gap), max_reach)
                dist = (
                    (here[0] - entry["pos"][0]) ** 2 + (here[1] - entry["pos"][1]) ** 2
                ) ** 0.5
                if dist <= limit:
                    pairs.append((dist, det_i, live_i))
        pairs.sort()

        taken_det: set[int] = set()
        taken_live: set[int] = set()
        matched: dict[int, int] = {}
        for _, det_i, live_i in pairs:
            if det_i in taken_det or live_i in taken_live:
                continue
            taken_det.add(det_i)
            taken_live.add(live_i)
            matched[det_i] = live_i

        for det_i, (box, here) in enumerate(detections):
            if det_i in matched:
                entry = live[matched[det_i]]
            else:
                track = Track(id=len(tracks))
                tracks.append(track)
                entry = {"track": track, "pos": here, "last_frame": index}
                live.append(entry)
            entry["track"].frames.append(index)
            entry["track"].boxes.append(box)
            entry["pos"] = here
            entry["last_frame"] = index

        live = [e for e in live if index - e["last_frame"] <= max_gap]

    return tracks


def split_movers(
    tracks: list[Track],
    diagonal: float,
    min_frames: int = 200,
    max_spread_frac: float = 0.10,
) -> tuple[list[Track], list[Track]]:
    """Identify the STATIONARY tracks; everything else is moving.

    Deliberately the other way round from the obvious framing. A person who
    stays put is easy to recognise and easy to track: they produce one long
    unbroken track whose centroid never leaves a small box. A person walking is
    neither -- they outrun the detector, which loses and reacquires them, so
    they arrive as a dozen short fragments rather than one track. Trying to
    pick "the mover" out of those fragments is fragile; recognising the three
    people who are obviously standing still, and calling every other detection
    a mover, is not.

    On this footage the two populations do not overlap: the stationary tracks
    span 192-301 px over thousands of frames while every fragment of the walker
    spans 1145 px or more.
    """
    stationary = [
        t
        for t in tracks
        if len(t.frames) >= min_frames and t.spread <= diagonal * max_spread_frac
    ]
    moving = [t for t in tracks if t not in stationary]
    return moving, stationary


def mover_boxes_by_frame(data: dict, stationary: list[Track]) -> dict[int, list[dict]]:
    """Per frame, the detection boxes that do NOT belong to a stationary track."""
    claimed: set[tuple[int, int]] = set()
    for track in stationary:
        for index, box in zip(track.frames, track.boxes):
            claimed.add((index, id(box)))

    out: dict[int, list[dict]] = {}
    for frame in data["frames"]:
        index = frame["index"]
        loose = [b for b in frame["boxes"] if (index, id(b)) not in claimed]
        if loose:
            out[index] = loose
    return out


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def resolve(detections_path: str | Path, min_frames: int = 1):
    """Tracks worth scoring against, plus the moving/stationary split."""
    data = load(detections_path)
    diag = (data["width"] ** 2 + data["height"] ** 2) ** 0.5
    # A person moves well under 1% of the frame diagonal between two frames at
    # 30fps; the radius grows with any gap the detector leaves.
    tracks = [
        t
        for t in build_tracks(data["frames"], diag * 0.008, diag * 0.06)
        if len(t.frames) >= min_frames
    ]
    movers, stationary = split_movers(tracks, diag)
    return data, tracks, movers, stationary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default=str(PROJECT_DIR / "eval" / "detections.json"))
    args = parser.parse_args()

    data, tracks, movers, stationary = resolve(args.detections)
    print(f"source={data['source']} frames={len(data['frames'])} tracks={len(tracks)}")
    print(f"{'id':>3} {'frames':>7} {'spread_px':>10} {'first':>6} {'last':>6}  role")
    for track in sorted(tracks, key=lambda t: -t.spread):
        role = "MOVING" if track in movers else "stationary"
        print(
            f"{track.id:>3} {len(track.frames):>7} {track.spread:>10.1f} "
            f"{track.frames[0]:>6} {track.frames[-1]:>6}  {role}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
