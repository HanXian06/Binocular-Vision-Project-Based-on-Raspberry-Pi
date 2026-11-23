"""选点应留在目标范围内，空间分散，且纯色画面不应产生虚构角点。"""
import unittest
import numpy as np
from stereo_ranging.selection import select_points

CFG=dict(quality_level=.01,min_distance=3,block_size=3,cells=3)

class SelectionTests(unittest.TestCase):
    def test_uniform_region_returns_no_points(self):
        im=np.full((80,120),100,np.uint8)
        self.assertEqual(len(select_points(im,[10,10,90,60],40,CFG)),0)

    def test_points_respect_roi_budget_and_spacing(self):
        rng=np.random.default_rng(12);im=rng.integers(0,256,(80,120),dtype=np.uint8)
        p=select_points(im,[10,12,90,60],30,CFG)
        self.assertGreater(len(p),10);self.assertLessEqual(len(p),30)
        self.assertTrue(((p[:,0]>=10)&(p[:,0]<100)&(p[:,1]>=12)&(p[:,1]<72)).all())
        distance=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(distance,np.inf)
        self.assertGreaterEqual(distance.min(),3)
        self.assertTrue(np.array_equal(p,select_points(im,[10,12,90,60],30,CFG)))

    def test_support_excludes_points(self):
        rng=np.random.default_rng(3);im=rng.integers(0,256,(80,120),dtype=np.uint8)
        support=np.ones(im.shape,bool);support[:,:60]=False
        p=select_points(im,[10,10,90,60],40,CFG,support)
        self.assertGreater(len(p),0);self.assertTrue((p[:,0]>=60).all())
        self.assertEqual(len(select_points(im,[10,10,90,60],0,CFG)),0)

if __name__=='__main__':unittest.main()
