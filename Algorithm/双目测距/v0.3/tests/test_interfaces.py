"""核对检测坐标还原、按类别去重，以及自动/人工选区使用同一统计流程。"""
import unittest
import numpy as np
from pathlib import Path
from detection.yolox_backend import Detector
from detection.yolox_backend import preprocess,decode
from pipeline import surface_box,measure_surface

class InterfaceTests(unittest.TestCase):
    def test_model_loads_from_workspace_chinese_path(self):
        path=Path(__file__).resolve().parents[1]/'models/yolox_nano.onnx'
        self.assertFalse(Detector(path).net.empty())

    def test_preprocessing_preserves_bgr_and_top_left_padding(self):
        image=np.full((480,640,3),(10,20,30),np.uint8)
        canvas,scale=preprocess(image)
        self.assertEqual(canvas.shape,(416,416,3));self.assertEqual(scale,.65)
        self.assertEqual(canvas[0,0].tolist(),[10,20,30])
        self.assertEqual(canvas[400,100].tolist(),[114,114,114])

    def test_decode_coordinates_and_confidence(self):
        pred=np.zeros((3549,85),np.float32)
        pred[0,:4]=[10,10,np.log(4),np.log(2)];pred[0,4]=.9;pred[0,5]=.8
        detections=decode(pred,(480,640),.35,.45)
        self.assertEqual(len(detections),1);d=detections[0]
        self.assertEqual(d['class_name'],'person');self.assertAlmostEqual(d['confidence'],.72,places=5)
        self.assertTrue(np.allclose(d['bbox_xyxy'],[64/.65,72/.65,96/.65,88/.65]))
        self.assertEqual(decode(pred,(480,640),.75,.45),[])

    def test_nms_is_per_class(self):
        p=np.zeros((3549,85),np.float32)
        for i in range(3):p[i,:4]=[10-i,10,np.log(4),np.log(2)];p[i,4]=.9
        p[0,5]=.9;p[1,5]=.8;p[2,5+56]=.9
        d=decode(p,(480,640),.35,.45)
        self.assertEqual({x['class_name'] for x in d},{'person','chair'});self.assertEqual(len(d),2)

    def test_surface_and_depth_rules(self):
        d=dict(class_name='person',bbox_xyxy=[100,50,300,450])
        box=surface_box(d,{'person':[.2,.15,.8,.65]},(640,480))
        self.assertEqual(box,[140,110,120,200])
        z=np.full((480,640),2.,np.float32)
        r=measure_surface(z,box,5)
        self.assertTrue(r['final']['available']);self.assertEqual(r['final']['depth_m'],2)
        self.assertEqual(len(r['regions']),13)
        r=measure_surface(np.full_like(z,np.nan),box,5)
        self.assertFalse(r['final']['available']);self.assertIsNone(r['final']['depth_m'])

if __name__=='__main__':unittest.main()
