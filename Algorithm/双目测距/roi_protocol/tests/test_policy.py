import unittest
import numpy as np
from policy import regions,contains,stats

class PolicyTests(unittest.TestCase):
    def test_regions_are_fixed_and_bounded(self):
        r=regions([-10,20,100,80]);self.assertEqual(len(r),13)
        self.assertEqual(r,regions([-10,20,100,80]))
        for x,y,w,h in r.values():self.assertTrue(x>=0 and y>=0 and x+w<=640 and y+h<=480)
    def test_missing_is_not_zero(self):
        s=stats(np.full(40,np.nan),np.ones(40,bool));self.assertIsNone(s['depth_m'])
        self.assertFalse(s['available'])
    def test_union_does_not_repeat_positions(self):
        p=np.array([[10,10],[15,15],[25,25]])
        mask=contains(p,[0,0,20,20])|contains(p,[5,5,20,20])
        self.assertEqual(int(mask.sum()),2)

if __name__=='__main__':unittest.main()
