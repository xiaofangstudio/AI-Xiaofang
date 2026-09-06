# -*- coding: utf-8 -*-
# v0.6 Flash 2 冒烟测试: 静默、快速、自动退出(不占内存)。
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v06flash2 as XF

# 静默快速: 隐藏深度思考打字 + 不思考(仅驱动回复/检索逻辑, 不联网)
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"

print("=== 实例化模型 (硬件自适应) ===")
xf = XF.XiaoFang()
print("版本=%s 引擎=%s | %s | d_model=%d L=%d H=%d vocab=%d KB=%d | GPU=%s" % (
    XF.VERSION, XF.ENGINE_NAME, XF.HW["gpu_model"], xf.transformer.d_model,
    xf.transformer.n_layers, xf.transformer.n_heads, xf.transformer.vocab_size,
    len(xf.retriever.docs), XF.HAS_GPU))

def get_answer(text):
    emo = xf.emotion.analyze(text)
    intent = xf.intent.detect(text, emo)
    kb = xf.retriever.retrieve(text)
    return xf.reply(text, emo, intent, kb)

ok = 0
def check(name, cond, extra=""):
    global ok
    if cond:
        ok += 1
        print("[PASS] %s %s" % (name, extra))
    else:
        print("[FAIL] %s %s" % (name, extra))

# —— 回归点: v0.6 Flash 2 ——
# 1) 官网
a = get_answer("给我小方工作室的网址")
check("官网问题返回站点",
      "fanggame.company" in a, "-> " + a[:60])

# 2) 身份/模型: Flash2 引擎 + 0.6 Flash 2 版本
a = get_answer("你的模型是什么")
check("模型名 Flash2 + 0.6 Flash 2",
      "Flash2" in a and "0.6 Flash 2" in a and "FlphaLit" not in a
      and "FangPro" not in a, "-> " + a[:70])

# 3) "图灵是谁" → 命中图灵人物条目 (flash2 词库尚无"图灵完备"条目, 那是正式版新增)
a = get_answer("图灵是谁")
check("图灵人物传记命中",
      "图灵" in a and ("数学" in a or "计算机" in a or "图灵机" in a or "人工智能" in a),
      "-> " + a[:80])

# 4) 词库 COMMON_WORDS 达标 (vocab 为 n-gram 训练后词集, 与正式版同源同量级)
import xiaofang_data_v06flash2 as _D
check("词库 COMMON_WORDS >= 8000", len(_D.COMMON_WORDS) >= 8000,
      "COMMON_WORDS=%d vocab=%d" % (len(_D.COMMON_WORDS), xf.transformer.vocab_size))

# 5) 智能闲聊关键词应答仍可用
a = get_answer("你好")
check("问候可答", bool(a) and len(a) > 2, "-> " + a[:40])

# 6) 命令联想表 / 线程超时工具
assert len(XF._COMMANDS) >= 4, "联想命令表缺失"
assert XF._run_with_timeout(lambda: 42, timeout=1.0) == 42, "线程超时工具异常"
check("命令联想表 + 超时工具", True)

print("=== 冒烟通过 %d/6 → 立即退出(释放内存) ===" % ok)
sys.stdout.flush()
os._exit(0)
