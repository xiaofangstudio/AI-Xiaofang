# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v04_beta as XF

print("=== 实例化模型 ===")
xf = XF.XiaoFang()
print("d_model =", xf.transformer.d_model)
print("n_layers =", xf.transformer.n_layers)
print("n_heads =", xf.transformer.n_heads)
print("vocab =", xf.transformer.vocab_size)
print("KB docs =", len(xf.retriever.docs))

tests = [
    "你是谁",
    "地球是什么？",
    "帮我介绍下Python",
    "我好累啊",
    "今天的日期是几号",
    "随便聊聊深度学习",
    "现在几点钟了",
    "给我讲个笑话",
    "谢谢你了",
    "我要怎么学Python编程",
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