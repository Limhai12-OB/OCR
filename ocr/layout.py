"""Reconstruct reading order and approximate columns from OCR box coordinates."""

from dataclasses import dataclass
from statistics import median

from ocr.postprocessing import clean_recognized_text


@dataclass
class TextDetection:
    """Recognized text with its original-image box and confidence."""

    text: str
    box: tuple[tuple[float, float], ...]
    confidence: float
    uncertain: bool

    @property
    def left(self) -> float:
        return min(point[0] for point in self.box)

    @property
    def top(self) -> float:
        return min(point[1] for point in self.box)

    @property
    def right(self) -> float:
        return max(point[0] for point in self.box)

    @property
    def bottom(self) -> float:
        return max(point[1] for point in self.box)

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)


def group_rows(detections: list[TextDetection]) -> list[list[TextDetection]]:
    """Group nearby boxes on the same baseline, retaining left-to-right order."""
    rows: list[dict] = []
    for detection in sorted(detections, key=lambda item: (item.center_y, item.left)):
        row = next(
            (
                candidate
                for candidate in reversed(rows)
                if abs(detection.center_y - candidate["center"])
                <= max(detection.height, candidate["height"]) * 0.55
            ),
            None,
        )
        if row is None:
            rows.append({"center": detection.center_y, "height": detection.height, "items": [detection]})
        else:
            row["items"].append(detection)
            row["center"] = sum(item.center_y for item in row["items"]) / len(row["items"])
            row["height"] = max(row["height"], detection.height)

    return [sorted(row["items"], key=lambda item: item.left) for row in rows]


def _character_width(detections: list[TextDetection]) -> float:
    estimates = [
        (item.right - item.left) / max(1, len(item.text.strip()))
        for item in detections
        if item.text.strip() and item.right > item.left
    ]
    return max(1.0, median(estimates)) if estimates else 8.0


def _format_row(row: list[TextDetection], char_width: float) -> str:
    """Use measured gaps between boxes to represent column separation."""
    output = ""
    previous_right = None
    for item in row:
        text = clean_recognized_text(item.text)
        if not text:
            continue
        if previous_right is not None:
            gap = max(0.0, item.left - previous_right)
            # Normal word gaps get one space; wider coordinate gaps become columns.
            # Keep large measured gaps; they often separate table columns.
            spaces = max(1, min(80, round(gap / char_width)))
            output += " " * spaces
        output += text
        previous_right = item.right
    return output.rstrip()


def reconstruct_text(detections: list[TextDetection], structured: bool = True) -> str:
    """Rebuild text in reading order, optionally preserving horizontal spacing."""
    rows = group_rows(detections)
    if not rows:
        return ""
    char_width = _character_width(detections)
    formatted = []
    previous_row = None
    for row in rows:
        line = _format_row(row, char_width)
        if not line:
            continue
        if structured and previous_row is not None:
            vertical_gap = min(item.top for item in row) - max(item.bottom for item in previous_row)
            typical_height = median([item.height for item in row + previous_row])
            if vertical_gap > typical_height * 1.2:
                formatted.append("")
        # Plain text keeps normal single-space reading while retaining line breaks.
        if not structured:
            line = " ".join(item.text.strip() for item in row if item.text.strip())
        formatted.append(line)
        previous_row = row
    return "\n".join(formatted).strip()
