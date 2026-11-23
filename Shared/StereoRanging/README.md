# 双目测距协作共享文件

更新者：陈珂乐，2026-11-23。此处是功能接口与交接材料，不含本地开发JSON约定。

- [接口约定JSON](interfaces.json)：检测对象、坐标、帧组、距离/状态、空值规则。
- [参数与坐标说明](PARAMETERS.md)：模型416输入、BGR与补边、640×480校正、米单位和参数可靠性。
- [版本对照及协作者路线](DEVELOPMENT_PATH.md)：已完成内容、漏检与测距问题、候选改进及验证顺序。
- [发布文件清单](MANIFEST.json)：本次发布副本的SHA-256；历史结果指纹与当前文件清单分开。

主要运行入口位于Algorithm/双目测距；修改后的YOLO接口位于Algorithm/Dual-Camera-YOLO/stereo_v0_3_interface，原识别文件不覆盖。
