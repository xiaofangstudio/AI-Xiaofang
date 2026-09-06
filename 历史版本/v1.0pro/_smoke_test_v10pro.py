# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.0 Pro —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0) 释放内存。重点验证"牛头不对马嘴"已修复。
import os
os.environ.setdefault("XF_QUIET", "1")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v10pro as M
M.SHOW_DEEP_THINK = False

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

_w("[i] 后端:", M._BACKEND)
_w("[i] 模型:", M.MODEL_TIER, "| 参数:", M.MODEL_PARAMS, "| 版本:", M.VERSION)

xf = M.XiaoFang()
_w("[ok] 装配完成")

_cases = [
    ("你好", None),
    ("你是谁", None),
    ("1+2等于几", None),
    ("小方工作室官网", None),
    ("什么是图灵完备", None),
    ("我想玩游戏，我有什么游戏推荐的吗？", "rec"),      # 关键回归: 必须给推荐, 不能答"游戏是什么"
    ("帮我推荐一部电影", "rec"),
    ("推荐系统是什么", "def"),                       # 反向: 这仍应走解释, 不能被推荐逻辑抢走
]
for c, tag in _cases:
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    out = xf.reply(c, emo, intent, kb) or ""
    assert out, "empty reply: " + c
    # 校验推荐/解释路由
    is_rec = xf._is_recommend(c)
    _genre = ("🔫", "🧭", "🎭", "🧩", "👥", "🎵", "🎬", "😂", "🚀", "🔍", "💛", "🤜", "📖", "🎧", "🍜", "🥘")
    if tag == "rec":
        assert is_rec, ("should be recommend", c, out[:40])
        assert any(g in out for g in _genre) and ("是什么" not in out[:12]), ("rec wrong content", c, out[:60])
    if tag == "def":
        assert not is_rec, ("should be definition, got recommend", c, out[:30])
    _w("[ok] 「", c, "」 -> ", (out or "").strip().split("\n")[0][:36])

_w("SMOKE_OK cases=%d" % len(_cases))
os._exit(0)