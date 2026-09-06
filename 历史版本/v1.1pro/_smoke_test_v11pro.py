# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.1 Pro —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0)。 重点验证 v1.1 Pro 修复:
#   "推荐要给具体东西, 不甩类型分类" + 原有"不曲解意图/不滥用知识库"全回归。
import os
os.environ.setdefault("XF_QUIET", "1")
import sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v11pro as M
M.SHOW_DEEP_THINK = False

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

_w("[i] 后端:", M._BACKEND, "| 版本:", M.VERSION)
assert M.VERSION == "1.1 Pro", ("version wrong", M.VERSION)
xf = M.XiaoFang()
_w("[ok] 装配完成")

def _ctx(c):
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    return emo, intent, kb, kb[0][0] if kb else 0.0

n = 0
# ---------- ① 该联网/该本地 路由决策 (纯判断, 不联网) ----------
online = ["我想要好的游戏推荐", "有什么好玩的游戏推荐吗", "传送门2通关攻略", "最近出了什么电影"]
local  = ["你是什么框架", "你的作者是谁", "你是谁用的什么做的", "谁开发的你", "你是谁"]
for c in online:
    emo, intent, kb, _ = _ctx(c)
    assert xf._needs_online(c, intent), ("should be online", c)
    _w("[ok] 该上网:", c); n += 1
for c in local:
    emo, intent, kb, _ = _ctx(c)
    assert not xf._needs_online(c, intent), ("should be local", c)
    _w("[ok] 该本地:", c); n += 1

# ---------- ② 问小方自己 → 身份本地答 (绝不"没能联网") ----------
id_cases = ["你是什么框架", "你的作者是谁", "你是谁"]
for c in id_cases:
    emo, intent, kb, _ = _ctx(c)
    out = xf.reply(c, emo, intent, kb) or ""
    assert out, "empty: " + c
    assert "没能联网" not in out and "这次没能" not in out, ("identity went to web", c, out[:40])
    assert ("小方" in out) or ("FlphaLit" in out) or ("工作室" in out), ("identity wrong", c, out[:40])
    _w("[ok] 身份本地答:", out.strip().split("\n")[0][:36]); n += 1

# ---------- ③ 传送门2 已入库 ----------
c = "传送门2是什么"
emo, intent, kb, kb_score = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert out and ("解谜" in out or "Valve" in out or "传送门" in out), ("portal def missing", out[:60])
_w("[ok] 传送门2是什么 -> ", out.strip().split("\n")[0][:40]); n += 1

c = "传送门2通关攻略"
emo, intent, kb, _ = _ctx(c)
assert xf._needs_online(c, intent), ("ridge should want web", c)
_val = M._kb_aligned(c, kb[0][1]) if kb else False
_w("[ok] 传送门2攻略 该联网=True(离线回退库概况) aligned=%s" % _val); n += 1

# ---------- ④ v1.1 Pro 核心: 推荐给"具体名字", 不再甩"类型分类" ----------
# 具体游戏名断言用清单首元素确保匹配
_games = [g[0] for g in xf._REC_GAMES]
c = "我想玩游戏，我有什么好玩的游戏推荐的吗？"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert xf._is_recommend(c), ("not recommend", c)
assert any(name in out for name in _games), ("rec no concrete game", c, out[:80])
assert "类型分成" not in out and "按口味分成" not in out, ("rec still type-split", out[:60])
_w("[ok] 游戏推荐给具体名: ", out.strip().split("\n")[0][:36]); n += 1

c = "推荐个好玩的游戏"
emo, intent, kb, _ = _ctx(c)
assert intent["top"] == "suggest", ("should be suggest", c, intent["top"])
_w("[ok] 推荐类意图=suggest (触发具体思考独白):", intent["top"]); n += 1

c = "有什么电影推荐"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
_movies = [m[0] for m in xf._REC_MOVIES]
assert any(name in out for name in _movies), ("rec no concrete movie", c, out[:80])
_w("[ok] 电影推荐给具体片名: ", out.strip().split("\n")[0][:36]); n += 1

# ---------- ⑤ 回归: 解释不走推荐/作文/写代码 ----------
c = "推荐系统是什么"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert not xf._is_recommend(c) and not xf._is_essay(c), ("def misrouted", c, out[:30])
_w("[ok] 推荐系统是什么 走定义: ", out.strip().split("\n")[0][:30]); n += 1

c = "帮我写一篇关于坚持的作文"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
body = out.split("\n\n", 1)[-1]
_nc = len(re.sub(r"\s", "", body))
assert xf._is_essay(c) and 400 <= _nc <= 900, ("essay routing/len", c, _nc)
_w("[ok] 作文 字数=", _nc); n += 1

c = "帮我写一个排序代码"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert not xf._is_essay(c), ("code->essay", out[:40])
_w("[ok] 写代码不走作文: ", out.strip().split("\n")[0][:24]); n += 1

_w("SMOKE_OK cases=%d" % n)
os._exit(0)