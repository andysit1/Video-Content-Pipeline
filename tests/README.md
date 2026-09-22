# Tests

The suite generates small real media with `ffmpeg` (a solid-color box on a
known audio pattern of tone/silence) and runs it through the real pipeline
code, rather than mocking ffmpeg/OpenCV out. That's what caught two of the
bugs fixed alongside these tests: `crop_image_crosshair()` swapped the
width/height axes (invisible on a square test array, obvious on a real
16:9-ish frame), and `analyze_silence()` mishandled a clip that ends mid-silence.

## Setup

```
pip install pytest opencv-python numpy ffmpeg-python icecream rich platformdirs
```

(or `poetry install --with dev` plus the project's own `requirements.txt`
deps). You also need the `ffmpeg` binary on PATH (`apt install ffmpeg` /
`brew install ffmpeg`).

Tests that need `ffmpeg`/`cv2`/`numpy`/`ffmpeg-python` skip automatically if
they're missing; the rest (regex parsing, ranking/threshold math, list
merging) run with no extra setup beyond `pytest` itself.

## Running

```
pytest
```

(`pyproject.toml` points pytest at `tests/` and puts `src/` on the path, so
this works from the repo root without an editable install.)

## Layout

- `conftest.py` -- shared fixtures: `fake_video` (a 4s clip with a known
  white box and a known tone/silence audio pattern), `make_color_clip`
  (factory for short solid-color clips), `fake_engine` (a minimal
  `PipelineEngine` stand-in), and availability skips.
- `test_audio_detection.py` -- silence/volume detection: real
  `silencedetect`/`volumedetect` output, the regex parsing in
  `analyze_data_stage.py`, and `get_silence_db()`'s threshold selection.
- `test_image_analysis.py` -- `OpenCVAggregate`: white-pixel percentage,
  crosshair cropping (both hand-built arrays and frames decoded from the
  generated video), threshold/edge detection.
- `test_clip_selection.py` -- `CompileVideoPipe`: ranking, thresholding,
  and an end-to-end run that concatenates real generated clips and checks
  the output's duration and frame content.
- `test_regressions.py` -- targeted tests for specific bugs fixed on this
  branch that aren't already covered end-to-end above.
