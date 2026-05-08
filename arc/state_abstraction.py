from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

from schemas.schemas import FrameAbstraction, ComponentInfo, TransitionAbstraction

FrameKey = Tuple[Tuple[int, ...], ...]

def frame_to_key(frame: List[List[int]]) -> FrameKey:
    return tuple(tuple(row) for row in frame)


def key_to_frame(frame_key: FrameKey) -> List[List[int]]:
    return [list(row) for row in frame_key]


def get_neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _validate_frame_key(frame_key: FrameKey) -> Tuple[int, int]:
    if not frame_key or not frame_key[0]:
        raise ValueError("Frame must be a non-empty 2D grid.")

    h = len(frame_key)
    w = len(frame_key[0])

    for row in frame_key:
        if len(row) != w:
            raise ValueError("Frame rows must all have the same length.")

    return h, w


def compute_value_counts(frame: List[List[int]]) -> Dict[int, int]:
    return compute_value_counts_from_key(frame_to_key(frame))


@lru_cache(maxsize=4096)
def compute_value_counts_from_key(frame_key: FrameKey) -> Dict[int, int]:
    counts = Counter()
    for row in frame_key:
        counts.update(row)
    return dict(counts)


def _compute_bbox_from_bounds(
    min_r: int, min_c: int, max_r: int, max_c: int
) -> Tuple[int, int, int, int]:
    return min_r, min_c, max_r, max_c


def extract_components(frame: List[List[int]], include_cells: bool = False) -> List[ComponentInfo]:
    return extract_components_from_key(frame_to_key(frame), include_cells=include_cells)


@lru_cache(maxsize=4096)
def _extract_components_without_cells(frame_key: FrameKey) -> Tuple[ComponentInfo, ...]:
    return _extract_components_impl(frame_key, include_cells=False)


@lru_cache(maxsize=512)
def _extract_components_with_cells(frame_key: FrameKey) -> Tuple[ComponentInfo, ...]:
    return _extract_components_impl(frame_key, include_cells=True)


def extract_components_from_key(frame_key: FrameKey, include_cells: bool = False) -> List[ComponentInfo]:
    if include_cells:
        return list(_extract_components_with_cells(frame_key))
    return list(_extract_components_without_cells(frame_key))


def _extract_components_impl(frame_key: FrameKey, include_cells: bool) -> Tuple[ComponentInfo, ...]:
    h, w = _validate_frame_key(frame_key)

    visited = [[False for _ in range(w)] for _ in range(h)]
    components: List[ComponentInfo] = []
    component_counter = 0

    for r in range(h):
        for c in range(w):
            if visited[r][c]:
                continue

            value = frame_key[r][c]
            q = deque()
            q.append((r, c))
            visited[r][c] = True

            size = 0
            sum_r = 0
            sum_c = 0
            min_r = max_r = r
            min_c = max_c = c
            touches_border = False
            cells: List[Tuple[int, int]] = [] if include_cells else []

            while q:
                cr, cc = q.popleft()

                size += 1
                sum_r += cr
                sum_c += cc
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True

                if include_cells:
                    cells.append((cr, cc))

                for nr, nc in get_neighbors(cr, cc, h, w):
                    if visited[nr][nc]:
                        continue
                    if frame_key[nr][nc] != value:
                        continue

                    visited[nr][nc] = True
                    q.append((nr, nc))

            bbox = _compute_bbox_from_bounds(min_r, min_c, max_r, max_c)
            centroid = (sum_r / size, sum_c / size)

            components.append(
                ComponentInfo(
                    component_id=f"comp_{component_counter}",
                    value=int(value),
                    size=size,
                    cells=cells,
                    bbox=bbox,
                    centroid=centroid,
                    touches_border=touches_border,
                    width=max_c - min_c + 1,
                    height=max_r - min_r + 1,
                )
            )
            component_counter += 1

    return tuple(components)


def extract_frame_abstraction(frame: List[List[int]], include_cells: bool = False) -> FrameAbstraction:
    return extract_frame_abstraction_from_key(frame_to_key(frame), include_cells=include_cells)


@lru_cache(maxsize=4096)
def _extract_frame_abstraction_without_cells(frame_key: FrameKey) -> FrameAbstraction:
    h, w = _validate_frame_key(frame_key)
    value_counts = compute_value_counts_from_key(frame_key)
    components = list(_extract_components_without_cells(frame_key))

    return FrameAbstraction(
        height=h,
        width=w,
        value_counts=value_counts,
        num_components=len(components),
        components=components,
    )


@lru_cache(maxsize=512)
def _extract_frame_abstraction_with_cells(frame_key: FrameKey) -> FrameAbstraction:
    h, w = _validate_frame_key(frame_key)
    value_counts = compute_value_counts_from_key(frame_key)
    components = list(_extract_components_with_cells(frame_key))

    return FrameAbstraction(
        height=h,
        width=w,
        value_counts=value_counts,
        num_components=len(components),
        components=components,
    )


def extract_frame_abstraction_from_key(frame_key: FrameKey, include_cells: bool = False) -> FrameAbstraction:
    if include_cells:
        return _extract_frame_abstraction_with_cells(frame_key)
    return _extract_frame_abstraction_without_cells(frame_key)


def _compute_changed_bbox(
    changed_positions: List[Tuple[int, int]]
) -> Optional[Tuple[int, int, int, int]]:
    if not changed_positions:
        return None

    rows = [r for r, _ in changed_positions]
    cols = [c for _, c in changed_positions]
    return min(rows), min(cols), max(rows), max(cols)


def _is_border_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _compute_border_change_stats(
    changed_positions: List[Tuple[int, int]],
    h: int,
    w: int,
) -> Tuple[int, float]:
    if not changed_positions:
        return 0, 0.0

    border_changed_cells = sum(
        1 for r, c in changed_positions if _is_border_cell(r, c, h, w)
    )
    border_changed_ratio = border_changed_cells / len(changed_positions)

    return border_changed_cells, border_changed_ratio


def _compute_significant_change(
    changed_cells: int,
    changed_ratio: float,
    border_changed_ratio: float,
) -> bool:
    if changed_cells == 0:
        return False

    if changed_cells >= 5:
        return True

    if changed_ratio >= 0.01 and border_changed_ratio < 0.8:
        return True

    return False


def extract_transition_abstraction(
    prev_frame: List[List[int]],
    curr_frame: List[List[int]],
) -> TransitionAbstraction:
    return extract_transition_abstraction_from_keys(
        frame_to_key(prev_frame),
        frame_to_key(curr_frame),
    )


@lru_cache(maxsize=8192)
def extract_transition_abstraction_from_keys(
    prev_frame_key: FrameKey,
    curr_frame_key: FrameKey,
) -> TransitionAbstraction:
    prev_h, prev_w = _validate_frame_key(prev_frame_key)
    curr_h, curr_w = _validate_frame_key(curr_frame_key)

    if (prev_h, prev_w) != (curr_h, curr_w):
        raise ValueError("Previous and current frames must have the same shape.")

    h, w = prev_h, prev_w

    changed_positions: List[Tuple[int, int]] = []
    changed_value_pairs: Dict[str, int] = {}

    for r in range(h):
        for c in range(w):
            a = prev_frame_key[r][c]
            b = curr_frame_key[r][c]
            if a != b:
                changed_positions.append((r, c))
                key = f"{a}->{b}"
                changed_value_pairs[key] = changed_value_pairs.get(key, 0) + 1

    changed_cells = len(changed_positions)
    changed_ratio = changed_cells / (h * w)
    changed_bbox = _compute_changed_bbox(changed_positions)
    border_changed_cells, border_changed_ratio = _compute_border_change_stats(
        changed_positions, h, w
    )

    significant_change = _compute_significant_change(
        changed_cells=changed_cells,
        changed_ratio=changed_ratio,
        border_changed_ratio=border_changed_ratio,
    )

    return TransitionAbstraction(
        changed_cells=changed_cells,
        changed_ratio=changed_ratio,
        changed_positions=changed_positions,
        changed_bbox=changed_bbox,
        changed_value_pairs=changed_value_pairs,
        border_changed_cells=border_changed_cells,
        border_changed_ratio=border_changed_ratio,
        significant_change=significant_change,
    )


def _summarize_components(components: List[ComponentInfo], max_components: int = 12) -> str:
    if not components:
        return "No components detected."

    sorted_components = sorted(
        components,
        key=lambda comp: comp.size,
        reverse=True,
    )

    lines = []
    for comp in sorted_components[:max_components]:
        lines.append(
            f"- {comp.component_id}: value={comp.value}, size={comp.size}, "
            f"bbox={comp.bbox}, centroid=({comp.centroid[0]:.1f}, {comp.centroid[1]:.1f}), "
            f"touches_border={comp.touches_border}, width={comp.width}, height={comp.height}"
        )

    if len(sorted_components) > max_components:
        lines.append(f"- ... {len(sorted_components) - max_components} more components omitted")

    return "\n".join(lines)


def summarize_for_llm(
    frame_abs: FrameAbstraction,
    trans_abs: Optional[TransitionAbstraction] = None,
) -> str:
    value_counts_str = ", ".join(
        f"{value}:{count}"
        for value, count in sorted(frame_abs.value_counts.items(), key=lambda x: x[0])
    )

    summary_parts = [
        "Structured state abstraction:",
        f"- grid_size: {frame_abs.height}x{frame_abs.width}",
        f"- value_counts: {{{value_counts_str}}}",
        f"- num_components: {frame_abs.num_components}",
        "- largest/components summary:",
        _summarize_components(frame_abs.components),
    ]

    if trans_abs is not None:
        summary_parts.extend([
            "",
            "Transition summary from previous frame to current frame:",
            f"- changed_cells: {trans_abs.changed_cells}",
            f"- changed_ratio: {trans_abs.changed_ratio:.6f}",
            f"- changed_bbox: {trans_abs.changed_bbox}",
            f"- changed_value_pairs: {trans_abs.changed_value_pairs}",
            f"- border_changed_cells: {trans_abs.border_changed_cells}",
            f"- border_changed_ratio: {trans_abs.border_changed_ratio:.6f}",
            f"- significant_change: {trans_abs.significant_change}",
        ])

    return "\n".join(summary_parts)