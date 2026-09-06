# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v05alpha as XF

print("=== 实例化模型 ===")
xf = XF.XiaoFang()
print("版本     =", XF.VERSION)
print("引擎名   =", XF.ENGINE_NAME)
print("d_model  =", xf.transformer.d_model)
print("n_layers =", xf.transformer.n_layers)
print("n_heads  =", xf.transformer.n_heads)
print("vocab    =", xf.transformer.vocab_size)
print("KB docs  =", len(xf.retriever.docs))
if hasattr(xf, "presets"):
    print("预设数   =", len(xf.presets))

tests = [
    "你是谁",
    "你是什么模型",
    "小方工作室官网是什么",
    "帮我介绍下Python",
    "量子引力最新研究是什么",
    "我好累啊",
    "今天的日期是几号",
    "hello",
    "こんにちは",
    "안녕하세요",
    "bonjour",
    "现在几点钟了",
    "给我讲个笑话",
    "谢谢你了",
    "什么是黑洞",
]
for t in tests:
    print("\n\n========== 输入: %s ==========" % t)
    try:
        xf.process_chat(t)
    except Exception:
        import traceback
        traceback.print_exc()

print("\n=== 冒烟测试结束 ===")