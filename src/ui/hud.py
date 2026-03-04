from __future__ import annotations

from moderngl_window.text.bitmapped import TextWriter2D


class HudOverlay:
    def __init__(self, font_size: float = 18.0, line_spacing: float = 4.0) -> None:
        self.writer = TextWriter2D()
        self.font_size = font_size
        self.line_spacing = line_spacing

    def draw(self, lines: list[str], viewport_height: int, x: float = 14.0, y_from_top: float = 18.0) -> None:
        y = float(viewport_height) - y_from_top
        for line in lines:
            self.writer.text = line
            self.writer.draw((x, y), size=self.font_size)
            y -= self.font_size + self.line_spacing

