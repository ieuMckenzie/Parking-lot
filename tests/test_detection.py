from backend.detection.detector import Detection


class TestDetectionProperties:
    def test_width(self):
        d = Detection(class_name="USDOT", bbox=(10, 20, 110, 70), confidence=0.9)
        assert d.width == 100

    def test_height(self):
        d = Detection(class_name="USDOT", bbox=(10, 20, 110, 70), confidence=0.9)
        assert d.height == 50


class TestDetectionPad:
    def test_pad_normal(self):
        d = Detection(class_name="USDOT", bbox=(100, 100, 200, 150), confidence=0.9)
        # width=100, height=50, pad 20% → pad_w=20, pad_h=10
        result = d.pad(0.2, (480, 640))
        assert result == (80, 90, 220, 160)

    def test_pad_clamps_to_zero(self):
        d = Detection(class_name="USDOT", bbox=(5, 3, 105, 53), confidence=0.9)
        # pad_w=20, pad_h=10 → x1=5-20=-15→0, y1=3-10=-7→0
        result = d.pad(0.2, (480, 640))
        assert result == (0, 0, 125, 63)

    def test_pad_clamps_to_frame_edge(self):
        d = Detection(class_name="USDOT", bbox=(580, 430, 640, 480), confidence=0.9)
        # width=60, height=50, pad_w=12, pad_h=10
        result = d.pad(0.2, (480, 640))
        assert result == (568, 420, 640, 480)

    def test_pad_zero_ratio(self):
        d = Detection(class_name="USDOT", bbox=(100, 100, 200, 150), confidence=0.9)
        result = d.pad(0.0, (480, 640))
        assert result == (100, 100, 200, 150)

    def test_pad_large_ratio(self):
        d = Detection(class_name="USDOT", bbox=(100, 100, 200, 150), confidence=0.9)
        # 100% pad: pad_w=100, pad_h=50
        result = d.pad(1.0, (480, 640))
        assert result == (0, 50, 300, 200)

    def test_pad_small_bbox_at_corner(self):
        d = Detection(class_name="LicensePlate", bbox=(0, 0, 20, 10), confidence=0.8)
        # width=20, height=10, pad_w=4, pad_h=2
        result = d.pad(0.2, (480, 640))
        assert result == (0, 0, 24, 12)
