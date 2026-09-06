# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.1 Alpha —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0) 释放内存。
# 重点验证: ①长线作文生成(500~800字, 整段完整稳定句子) ②推荐修复仍生效 ③成文不误抢"写代码"
import os
os.environ.setdefault("XF_QUIET", "1")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v11alpha as M
M.SHOW_DEEP_THINK = False

PUNCT = ("。", "！", "？", "～")

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

def _cz(s):
    return len(re_sub_ws(s))

import re
def re_sub_ws(s):
    return re.sub(r"\s", "", s)

_w("[i] 后端:", M._BACKEND)
_w("[i] 模型:", M.MODEL_TIER, "| 参数:", M.MODEL_PARAMS, "| 版本:", M.VERSION)
assert M.VERSION == "1.1 Alpha", ("version wrong", M.VERSION)

xf = M.XiaoFang()
_w("[ok] 装配完成")

# ---------- 长线作文回归 ----------
_essay_cases = [
    "帮我写一篇关于坚持的作文",
    "写一篇关于友谊的作文",
    "以我的梦想为题写一篇作文",
    "围绕'时间'写一段感想",
]
for c in _essay_cases:
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    out = xf.reply(c, emo, intent, kb) or ""
    assert out, "empty essay: " + c
    assert xf._is_essay(c), ("essay not routed", c)
    body = out.split("\n\n", 1)[-1]
    n = _cz(body)
    assert 400 <= n <= 900, ("essay length out of range", c, n)
    # 每个句子都应以标点结束 —— 不允许出现"断句/胡言乱语"
    for sent in [s for s in re.split(r"(?<=[。！？～])", body) if s.strip()]:
        if len(sent.strip()) >= 1 and not sent.strip().endswith(PUNCT) and not sent.strip().endswith("。"):
            raise AssertionError(("dangling sentence", c, sent))
    _w("[ok] 作文 「", c, "」 字数=", n)

# ---------- 反向: 写代码不应走作文 ----------
c = "帮我写一个排序代码"
emo = xf.emotion.analyze(c)
intent = xf.intent.detect(c, emo)
kb = xf.retriever.retrieve(c)
out = xf.reply(c, emo, intent, kb) or ""
assert not xf._is_essay(c), ("code mis-routed to essay", out[:40])
assert ("排序" in out) or ("代码" in out) or ("def " in out), ("code reply missing", out[:60])
_w("[ok] 写代码不走作文: ", out.strip().split("\n")[0][:36])

# ---------- 普通/推荐/解释回归 ----------
_cases = [
    ("你好", None),
    ("你是谁", None),
    ("1+2等于几", None),
    ("小方工作室官网", None),
    ("什么是图灵完备", None),
    ("我想玩游戏，我有什么游戏推荐的吗？", "rec"),
    ("帮我推荐一部电影", "rec"),
    ("推荐系统是什么", "def"),
]
for c, tag in _cases:
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    out = xf.reply(c, emo, intent, kb) or ""
    assert out, "empty reply: " + c
    is_rec = xf._is_recommend(c)
    assert not xf._is_essay(c), ("non-essay routed to essay", c, out[:40])
    _genre = ("🔫", "🧭", "🎭", "🧩", "👥", "🎵", "🎬", "😂", "🚀", "🔍", "💛", "🤜", "📖", "🎧", "🍜", "🥘")
    if tag == "rec":
        assert is_rec, ("should be recommend", c, out[:40])
        assert any(g in out for g in _genre) and ("是什么" not in out[:12]), ("rec wrong content", c, out[:60])
    if tag == "def":
        assert not is_rec, ("should be definition, got recommend", c, out[:30])
    _w("[ok] 「", c, "」 -> ", (out or "").strip().split("\n")[0][:36])

_w("SMOKE_OK cases=%d essays=%d" % (len(_cases) + len(_essay_cases) + 1, len(_essay_cases)))
os._exit(0)