"""测距的必要数值检查。运行：python -m unittest discover -s tests -v。"""
import unittest
import numpy as np
from stereo_ranging.matching import compute_disparity, support_mask
from stereo_ranging.geometry import depth_from_disparity
from stereo_ranging.region import measure_region

class NumericTests(unittest.TestCase):
    def test_constant_disparity_in_meters(self):
        """已知Z=fB/d，100×0.0585/8=0.73125米，用它核对Q重建的单位。"""
        q=np.array([[1,0,0,-10],[0,1,0,-10],[0,0,0,100],[0,0,1/.0585,0]],np.float64)
        z=depth_from_disparity(np.full((20,20),8,np.float32),q)
        self.assertTrue(np.allclose(z,.73125))
    def test_region_excludes_invalid_and_reports_unavailable(self):
        """ROI中只有1和1.1有效，中位数为1.05；提高数量门槛后应返回空距离。"""
        z=np.array([[1,1.1],[np.nan,-3]],np.float32)
        r=measure_region(z,np.ones((2,2),bool),(0,0,2,2),2,.5)
        self.assertAlmostEqual(r['depth_m'],1.05,places=5)
        self.assertEqual(r['valid_pixels'],2)
        r=measure_region(z,np.ones((2,2),bool),(0,0,2,2),3,.5)
        self.assertIsNone(r['depth_m'])
        self.assertFalse(r['available'])
    def test_right_support_uses_corresponding_x(self):
        """视差为2时，左图x=3对应右图x=1；右图这个位置无效，应排除左点。"""
        d=np.full((3,8),2,np.float32)
        left=np.ones(d.shape,bool)
        right=left.copy()
        right[:,1]=False
        m=support_mask(d,left,right,1)
        self.assertFalse(m[1,3])
        self.assertTrue(m[1,4])
        self.assertFalse(m[1,0])
    def test_sgbm_disparity_sign_and_scale(self):
        """将右图内容向左移8像素，验证匹配得到正8像素及正确的16倍换算。"""
        # 固定随机种子，使合成纹理每次相同，便于复现测试。
        rng=np.random.default_rng(1)
        l=rng.integers(0,256,(96,320),dtype=np.uint8)
        r=np.zeros_like(l)
        r[:,:-8]=l[:,8:]
        raw,d=compute_disparity(l,r,{'min_disparity':0,'num_disparities':128,'block_size':5,'p1':200,'p2':800,'uniqueness_ratio':10,'speckle_window_size':0,'speckle_range':2,'disp12_max_diff':1})
        # 去掉上下边缘和左侧搜索边界，在内部区域检查视差中位数。
        self.assertAlmostEqual(float(np.median(d[15:-15,150:280])),8,delta=.2)
        self.assertTrue(np.allclose(d,raw.astype(np.float32)/16))
    def test_roi_bounds(self):
        """超出图像范围的ROI应明确报错，避免实际统计区域与选框不一致。"""
        with self.assertRaises(ValueError):measure_region(np.ones((5,5)),np.ones((5,5),bool),(4,4,3,3),1,.1)

if __name__=='__main__':
    unittest.main()
