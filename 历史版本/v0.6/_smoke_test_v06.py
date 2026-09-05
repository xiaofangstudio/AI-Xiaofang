# -*- coding: utf-8 -*-
# v0.6 正式版 冒烟测试: 静默、快速、自动退出(不占内存)。
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v06 as XF

# 静默快速: 隐藏深度思考打字 + 不思考(仅驱动回复/检索逻辑, 不联网)
_OLD_SHOW = XF.SHOW_DEEP_THINK
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"

print("=== 实例化模型 (硬件自适应) ===")
xf = XF.XiaoFang()
print("版本=%s 引擎=%s | %s | d_model=%d L=%d H=%d vocab=%d KB=%d | GPU=%s 思考=%s" % (
    XF.VERSION, XF.ENGINE_NAME, XF.HW["gpu_model"], xf.transformer.d_model,
    xf.transformer.n_layers, xf.transformer.n_heads, xf.transformer.vocab_size,
    len(xf.retriever.docs), XF.HAS_GPU, XF.THINK_MODE))

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

# —— 关键回归: 修复点 v0.6 正式版 ——
# 1) 官网 + 版本号
a = get_answer("给我小方工作室的网址")
check("官网问题返回站点+版本",
      "fanggame.company" in a and "0.6 正式版" in a, "-> " + a[:60])

# 2) 身份/模型: 新名 FangPro, 无旧名 FlphaLit/Flash2, 带版本
a = get_answer("你的模型是什么")
check("模型名 FangPro + 版本 + 无旧名",
      "FangPro" in a and "0.6 正式版" in a and "FlphaLit" not in a
      and "Flash2" not in a and "Flpalit" not in a, "-> " + a[:60])

# 3) 图灵完备 → 答"完备标准", 不是"介绍图灵是谁"
a = get_answer("图灵完备的标准是什么")
check("图灵完备命中正确概念(非图灵人传)",
      "图灵完备" in a and ("循环" in a or "条件" in a or "分支" in a
                            or "存储" in a), "-> " + a[:80])

# 4) emoji 只出现句首/句尾, 不打断句中
mid_chars = set("😂😊👍🌟✨💡🚀🔒🌐🕐📅🎈🧠🎯✏️")
sample = "这是一段用于验证恢复正常的长文本内容，它足够长好用来检查中间位置是否混入表情。"
def has_mid_emoji(r):
    for i, ch in enumerate(r):
        if ch in mid_chars:
            # 允许: 句首(索引0或1, 可能带空格) / 句尾(末位)
            if not (i <= 1 or i >= len(r) - 1):
                return True
    return False
all_clean = True
for _ in range(80):
    r = xf.generator._insert_emojis(sample, {"score": 5, "intensity": 9})
    if has_mid_emoji(r):
        all_clean = False
        break
check("emoji 不插句中(仅句首/句尾)", all_clean)

# 5) 智能闲聊关键词应答仍可用
a = get_answer("你好")
check("问候可答", bool(a) and len(a) > 2, "-> " + a[:40])

# 6) 命令联想表 / 线程超时工具
assert len(XF._COMMANDS) >= 4, "联想命令表缺失"
assert XF._run_with_timeout(lambda: 42, timeout=1.0) == 42, "线程超时工具异常"
check("命令联想表 + 超时工具", True, "%s" % XF._COMMANDS)

print("=== 冒烟通过 %d/%d → 立即退出(释放内存) ===" % (ok, 6))
sys.stdout.flush()
# 立即结束, 不残留后台线程/不占内存
os._exit(0)