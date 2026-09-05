# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_data as DATA

print("=== 数据检查 ===")
print("COMMON_WORDS 数量:", len(DATA.COMMON_WORDS))
print("情感正面词:", len(DATA.EMOTION_POS))
print("情感负面词:", len(DATA.EMOTION_NEG))
print("知识库条目:", len(DATA.KNOWLEDGE_BASE))

import xiaofang as XF

print("\n=== 实例化模型 ===")
xf = XF.XiaoFang()
print("版本     =", XF.VERSION)
print("引擎名   =", XF.ENGINE_NAME)
print("d_model  =", xf.transformer.d_model)
print("n_layers =", xf.transformer.n_layers)
print("n_heads  =", xf.transformer.n_heads)
print("vocab    =", xf.transformer.vocab_size)
print("KB docs  =", len(xf.retriever.docs))

tests = [
    "你好",
    "你是谁",
    "什么是人工智能",
    "我好累",
]
for t in tests:
    print("\n\n========== 输入: %s ==========" % t)
    try:
        xf.process_chat(t)
    except Exception:
        import traceback
        traceback.print_exc()

print("\n=== 冒烟测试结束 ===")
