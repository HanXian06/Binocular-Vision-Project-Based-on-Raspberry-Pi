"""左右对应检查的数值测试：重点核对符号、实际对应坐标和无效邻点。"""
import unittest

import numpy as np

from stereo_ranging.consistency import compute_right_disparity, left_right_consistency


class ConsistencyTests(unittest.TestCase):
    def test_reverse_match_of_eight_pixel_shift(self):
        """左图向右比右图多8像素时，右图反查结果应为−8，而不是+8。"""
        rng = np.random.default_rng(2)
        left = rng.integers(0, 256, (96, 320), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        cfg = dict(min_disparity=0, num_disparities=128, block_size=5,
            p1=200, p2=800, uniqueness_ratio=10, speckle_window_size=0,
            speckle_range=2, disp12_max_diff=1)
        support = np.ones(left.shape, bool)
        raw, disparity, valid = compute_right_disparity(left, right, (support, support), cfg)
        self.assertAlmostEqual(float(np.median(disparity[15:-15, 40:150])), -8, delta=.2)
        self.assertGreater(float(valid[15:-15, 40:150].mean()), .95)
        self.assertFalse(valid[:, -1].any())
        self.assertTrue(np.allclose(disparity, raw.astype(np.float32)/16))
        self.assertEqual(cfg['min_disparity'], 0)

    def test_uses_match_coordinate_and_rejects_wrong_return(self):
        """左x=4、d=2应查右x=2；右视差−5会回到左x=7，往返误差为3。"""
        left = np.full((2, 8), 2, np.float32)
        right = np.full_like(left, -2)
        right[:, 2] = -5
        valid = np.ones(left.shape, bool)
        result = left_right_consistency(left, right, valid, valid, 1)
        self.assertTrue(result['consistent'][0, 5])
        self.assertFalse(result['consistent'][0, 4])
        self.assertEqual(float(result['residual_px'][0, 4]), 3)
        self.assertFalse(result['checked'][0, 0])
        self.assertTrue(np.isnan(result['residual_px'][0, 0]))

    def test_fractional_sample_requires_both_valid_neighbors(self):
        """右坐标为1.5时，对−2和−3插值得−2.5；任一邻点无效则不检查该点。"""
        left = np.full((2, 8), 2.5, np.float32)
        right = np.full_like(left, -2.5)
        right[:, 1], right[:, 2] = -2, -3
        valid = np.ones(left.shape, bool)
        result = left_right_consistency(left, right, valid, valid, .1)
        self.assertTrue(result['consistent'][0, 4])
        self.assertEqual(float(result['sampled_right_px'][0, 4]), -2.5)
        right_valid = valid.copy()
        right_valid[:, 2] = False
        result = left_right_consistency(left, right, valid, right_valid, .1)
        self.assertFalse(result['checked'][0, 4])
        self.assertFalse(result['consistent'][0, 4])

    def test_integer_sample_does_not_require_unused_neighbor(self):
        """恰好查右x=2时，只需要x=2有效，不应错误地要求x=3也有效。"""
        left = np.full((2, 8), 2, np.float32)
        right = -left.copy()
        valid = np.ones(left.shape, bool)
        rv = valid.copy()
        rv[:, 3] = False
        result = left_right_consistency(left, right, valid, rv, 1)
        self.assertTrue(result['consistent'][0, 4])

    def test_nonfinite_or_invalid_left_is_not_checked(self):
        """无效左点、NaN和右侧非有限值不应进入统计，也不改写输入数组。"""
        left = np.full((2, 8), 2, np.float32)
        right = -left.copy()
        left[0, 4] = np.nan
        right[0, 3] = np.inf
        valid = np.ones(left.shape, bool)
        lv = valid.copy()
        lv[1, 4] = False
        result = left_right_consistency(left, right, lv, valid, 1)
        self.assertFalse(result['checked'][0, 4])
        self.assertFalse(result['checked'][0, 5])
        self.assertFalse(result['checked'][1, 4])
        self.assertTrue(np.isnan(left[0, 4]))
        self.assertTrue(np.isinf(right[0, 3]))
        with self.assertRaises(ValueError):
            left_right_consistency(left, right, valid, valid, -1)


if __name__ == '__main__':
    unittest.main()
