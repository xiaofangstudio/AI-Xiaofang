# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v05alpha2 as XF

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
if hasattr(xf, "persona"):
    print("人格库   =", len(xf.persona.entries))

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
    # --- v0.5 Alpha 2 人格触发(问生活/身世) ---
    "你家住在哪里",
    "你爸爸妈妈叫什么",
    "你几岁啦",
    "你长什么样",
    "你喜欢吃什么",
    "你有什么宠物",
    "你的围巾是怎么回事",
    "天空王国发生了什么事",
    "你最好的朋友是谁",
    # --- 中性问题不得带人格出现 ---
    "什么是编程",
    "讲讲Python的一个知识点",
]
for t in tests:
    print("\n\n========== 输入: %s ==========" % t)
    try:
        xf.process_chat(t)
    except Exception:
        import traceback
        traceback.print_exc()

print("\n=== 冒烟测试结束 ===")