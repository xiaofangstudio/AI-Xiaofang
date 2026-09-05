# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.0 正式版 —— 快速冒烟测试
# 遵循项目规范: 静默(XF_QUIET)、关闭深层思考打印、跑完 os._exit(0) 立即退出并释放内存。
import os
os.environ.setdefault("XF_QUIET", "1")       # 隐藏启动进度(列表符/百分比)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v10 as M
M.SHOW_DEEP_THINK = False                    # 冒烟测试静默思考

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

_w("[i] 后端:", M._BACKEND)
_w("[i] 模型:", M.MODEL_TIER, "| 参数:", M.MODEL_PARAMS)

xf = M.XiaoFang()
_w("[ok] 装配完成")

cases = [
    "你好",
    "你是谁",
    "1+2等于几",
    "小方工作室官网",
    "什么是图灵完备",
]
for c in cases:
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    kb_score = kb[0][0] if kb else 0.0
    kb_top = kb[0][1]["t"] if kb else None
    mono = xf._think_monologue(c, emo, intent, kb, kb_top, kb_score, [], [], False)
    out = xf.reply(c, emo, intent, kb)
    assert mono and out, "case failed: " + c
    _w("[ok] 输入「", c, "」 -> 独白段数", len(mono), "| 回答:", (out or "").strip()[:24])

_w("SMOKE_OK cases=%d" % len(cases))
os._exit(0)