# -*- coding: utf-8 -*-
# v0.6 Pro 冒烟测试: 静默、快速、自动退出(不占内存)。
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v06pro as XF

# 静默快速: 隐藏深度思考打字 + 不思考(仅驱动回复/检索/生成逻辑, 不联网)
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"

print("=== 实例化模型 (硬件自适应, 纯自研 GPU/CPU 后端) ===")
xf = XF.XiaoFang()
print("版本=%s 引擎=%s | 后端=%s | %s | d_model=%d L=%d H=%d vocab=%d KB=%d | GPU=%s" % (
    XF.VERSION, XF.ENGINE_NAME, XF._BACKEND, XF.HW["gpu_model"], xf.transformer.d_model,
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

# 0) 词库存量达标 (>= 8000)
check("词库 >= 8000", xf.transformer.vocab_size >= 8000,
      "vocab=%d" % xf.transformer.vocab_size)

# 1) 要网址只给网址, 不介绍自己/工作室
a = get_answer("给我小方工作室的网址")
check("网址就只给网址",
      "fanggame.company" in a and ("我是" not in a and "模型名" not in a and "出品" not in a
                                    and "版本" not in a), "-> " + a)

# 2) 身份/模型: 新名 FangPro + 0.6 Pro, 无旧名
a = get_answer("你的模型是什么")
check("模型名 FangPro + 0.6 Pro + 无旧名",
      "FangPro" in a and "0.6 Pro" in a and "FlphaLit" not in a
      and "Flash2" not in a and "正式版" not in a, "-> " + a[:50])

# 3) 图灵完备仍答正确概念(回归)
a = get_answer("图灵完备的标准是什么")
check("图灵完备命中正确概念",
      "图灵完备" in a and ("循环" in a or "条件" in a or "分支" in a or "存储" in a),
      "-> " + a[:70])

# 4) TF trace 优化后无异常(GPU/CPU 双后端安全): forward 返回结构完整
probs, tr = xf.transformer.forward([xf.transformer.token2id.get(t, 0)
                                    for t in XF.Tokenizer().tokenize("你好小方")][:24] or [0], trace=True)
check("TF forward + trace 双后端稳定",
      len(tr["blocks"]) == xf.transformer.n_layers and tr["probs"] is not None)
del probs

# 5) 命令联想表 + 线程超时工具
assert len(XF._COMMANDS) >= 4, "联想命令表缺失"
assert XF._run_with_timeout(lambda: 42, timeout=1.0) == 42, "线程超时工具异常"
check("命令联想表 + 超时工具", True)

print("=== 冒烟通过 %d/6 → 立即退出(释放内存) ===" % ok)
sys.stdout.flush()
os._exit(0)