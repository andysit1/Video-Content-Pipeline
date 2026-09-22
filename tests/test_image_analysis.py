"""
Video-image analysis tests: build frames with a known amount of "white" in
them (a centered box of a known size) and check that the OpenCV aggregate
functions used by AnalyzeClipsPipe compute the expected percentages -- both
for hand-built numpy frames and for frames decoded out of a real generated
video clip.
"""
import numpy as np
import pytest

from VidFlow.aggregate.opencv_component import OpenCVAggregate
from VidFlow.aggregate.utils import percentage_of_white_pixels


def make_boxed_frame(size=300, box=100, value=255):
    frame = np.zeros((size, size), dtype=np.uint8)
    start = (size - box) // 2
    end = start + box
    frame[start:end, start:end] = value
    return frame


class TestPercentageOfWhitePixels:
    def test_all_black(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        assert percentage_of_white_pixels(frame) == 0.0

    def test_all_white(self):
        frame = np.full((10, 10), 255, dtype=np.uint8)
        assert percentage_of_white_pixels(frame) == 100.0

    def test_quarter_white(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        frame[:5, :5] = 255  # 25 of 100 pixels
        assert percentage_of_white_pixels(frame) == pytest.approx(25.0)

    def test_rejects_color_image(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            percentage_of_white_pixels(frame)


class TestCropImageCrosshair:
    def test_crops_centered_region_of_expected_size(self):
        agg = OpenCVAggregate()
        frame = make_boxed_frame(size=300, box=100)
        cropped = agg.crop_image_crosshair(frame)

        offset = agg.crosshair_offset
        assert cropped is not False
        assert cropped.shape == (offset * 2, offset * 2)

    def test_generated_white_box_is_fully_captured_by_crop(self):
        agg = OpenCVAggregate()
        box = 100
        frame = make_boxed_frame(size=300, box=box)
        cropped = agg.crop_image_crosshair(frame)

        expected_white_pixels = box * box
        expected_pct = expected_white_pixels / (cropped.shape[0] * cropped.shape[1]) * 100
        assert percentage_of_white_pixels(cropped) == pytest.approx(expected_pct)

    def test_returns_false_for_frame_smaller_than_offset(self):
        agg = OpenCVAggregate()
        tiny = np.zeros((50, 50), dtype=np.uint8)
        assert agg.crop_image_crosshair(tiny) is False


class TestThresholdAndEdgeDetection:
    def test_do_binary_threshold_matches_known_white_percentage(self):
        agg = OpenCVAggregate()
        box = 100
        frame = make_boxed_frame(size=300, box=box)
        cropped = agg.crop_image_crosshair(frame)

        expected_pct = (box * box) / (cropped.shape[0] * cropped.shape[1]) * 100
        assert agg.do_binary_threshold(cropped) == pytest.approx(expected_pct)

    def test_canny_edges_are_sparser_than_the_filled_region(self):
        agg = OpenCVAggregate()
        frame = make_boxed_frame(size=300, box=100)
        cropped = agg.crop_image_crosshair(frame)

        filled_pct = agg.do_binary_threshold(cropped)
        edge_pct = agg.get_canny_edge_detection_white_percentage(cropped)

        assert 0 < edge_pct < filled_pct


def test_analysis_on_frames_decoded_from_a_generated_video(fake_video):
    import cv2

    agg = OpenCVAggregate()
    cap = cv2.VideoCapture(str(fake_video["path"]))
    ret, frame = cap.read()
    cap.release()
    assert ret, "failed to decode a frame from the generated fixture video"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped = agg.crop_image_crosshair(gray)
    assert cropped is not False

    box = fake_video["box_size"]
    offset = agg.crosshair_offset
    expected_pct = (box * box) / ((offset * 2) ** 2) * 100

    actual_pct = agg.do_binary_threshold(cropped)
    # small tolerance for h.264 encoding softening the box edges a bit,
    # unlike the exact match the synthetic-array tests can demand.
    assert actual_pct == pytest.approx(expected_pct, abs=3)
