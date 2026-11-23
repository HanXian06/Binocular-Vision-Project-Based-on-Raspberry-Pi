"""调用独立测距包的离线入口，不改变原双摄识别程序。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Algorithm/双目测距/v0.3'))
from run_offline import main
if __name__=='__main__':main()
