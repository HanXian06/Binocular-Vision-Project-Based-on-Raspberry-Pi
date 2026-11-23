"""核对局部匹配方向、候选歧义与三角化；期望值由合成位移和几何直接给出。"""
import unittest
import numpy as np
from stereo_ranging.local_matching import match_points, triangulate_matches

class LocalTests(unittest.TestCase):
    def test_known_horizontal_and_vertical_shift(self):
        rng=np.random.default_rng(41)
        left=rng.integers(0,256,(70,180),dtype=np.uint8)
        right=np.zeros_like(left)
        right[3:,:-8]=left[:-3,8:]
        result=match_points(left,right,[(90,30)],vertical_radius=5)
        self.assertTrue(result[0]['accepted'])
        self.assertAlmostEqual(result[0]['right_xy'][0],82,delta=.2)
        self.assertEqual(result[0]['right_xy'][1],33)
        self.assertGreater(result[0]['score'],.99)

    def test_uniform_patch_rejected(self):
        im=np.full((50,180),100,np.uint8)
        result=match_points(im,im,[(90,25)],vertical_radius=5)
        self.assertFalse(result[0]['accepted'])
        self.assertEqual(result[0]['reason'],'weak_texture')

    def test_repeated_identical_candidates_rejected(self):
        rng=np.random.default_rng(3)
        left=rng.integers(0,256,(60,180),dtype=np.uint8)
        right=rng.integers(0,256,left.shape,dtype=np.uint8)
        patch=left[26:35,96:105]
        right[26:35,76:85]=patch
        right[26:35,56:65]=patch
        row=match_points(left,right,[(100,30)],vertical_radius=0)[0]
        self.assertFalse(row['accepted'])
        self.assertEqual(row['reason'],'ambiguous')

    def test_invalid_source_support_rejected(self):
        rng=np.random.default_rng(4)
        a=rng.integers(0,256,(60,180),dtype=np.uint8)
        mask=np.ones(a.shape,bool);mask[30,90]=False
        row=match_points(a,a,[(90,30)],vertical_radius=0,supports=(mask,mask))[0]
        self.assertEqual(row['reason'],'unsupported_left')

    def test_triangulation_and_vertical_incompatibility(self):
        p1=np.array([[100,0,50,0],[0,100,30,0],[0,0,1,0]],float)
        p2=p1.copy();p2[0,3]=-5.85
        xyz,error=triangulate_matches(np.array([[60,30],[60,30]],float),np.array([[52,30],[52,34]],float),p1,p2)
        self.assertAlmostEqual(xyz[0,2],.73125,places=6)
        self.assertLess(error[0],1e-5)
        self.assertGreater(error[1],1)

if __name__=='__main__':unittest.main()
