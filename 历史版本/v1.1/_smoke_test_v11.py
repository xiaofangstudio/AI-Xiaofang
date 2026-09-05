# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.1 正式版 —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0)。 重点验证"不曲解意图/不滥用知识库":
#   ①该上网搜(游戏推荐/攻略)与该本地答(问小方自己)的路由决策 ②身份一律本地答 ③传送门2入库 ④作文/推荐/写代码回归
import os
os.environ.setdefault("XF_QUIET", "1")
import sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v11 as M
M.SHOW_DEEP_THINK = False

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

_w("[i] 后端:", M._BACKEND, "| 版本:", M.VERSION)
assert M.VERSION == "1.1", ("version wrong", M.VERSION)
xf = M.XiaoFang()
_w("[ok] 装配完成")

def _ctx(c):
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    return emo, intent, kb, kb[0][0] if kb else 0.0

# ---------- ① 该联网/该本地 路由决策 (纯判断, 不联网) ----------
online = ["我想要好的游戏推荐", "有什么好玩的游戏推荐吗", "传送门2通关攻略", "最近出了什么电影"]
local  = ["你是什么框架", "你的作者是谁", "你是谁用的什么做的", "谁开发的你", "你是谁"]
for c in online:
    emo, intent, kb, _ = _ctx(c)
    assert xf._needs_online(c, intent), ("should be online", c)
    _w("[ok] 该上网:", c)
for c in local:
    emo, intent, kb, _ = _ctx(c)
    assert not xf._needs_online(c, intent), ("should be local", c)
    _w("[ok] 该本地:", c)

# ---------- ② 问小方自己 → 身份本地答 (绝不出现"没能联网") ----------
id_cases = ["你是什么框架", "你的作者是谁", "你是谁"]
for c in id_cases:
    emo, intent, kb, _ = _ctx(c)
    out = xf.reply(c, emo, intent, kb) or ""
    assert out, "empty: " + c
    assert "没能联网" not in out and "这次没能" not in out, ("identity went to web", c, out[:40])
    assert ("小方" in out) or ("FlphaLit" in out) or ("工作室" in out) or ("模型" in out), ("identity wrong", c, out[:40])
    _w("[ok] 身份本地答:", out.strip().split("\n")[0][:40])

# ---------- ③ 传送门2 已入库 (离线时"攻略"回退给知识库概况, 非空且稳定; "是什么"走本地定义) ----------
c = "传送门2是什么"
emo, intent, kb, kb_score = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert out and ("解谜" in out or "Valve" in out or "传送门" in out), ("portal def missing", out[:60])
_w("[ok] 传送门2是什么 -> ", out.strip().split("\n")[0][:40])

c = "传送门2通关攻略"
emo, intent, kb, _ = _ctx(c)
assert xf._needs_online(c, intent) or xf._should_web(c, emo, intent, kb), ("ridge should want web", c)
_val = M._kb_aligned(c, kb[0][1]) if kb else False
_w("[ok] 传送门2攻略 该联网=True(离线回退库概况) aligned=%s" % _val)

# ---------- ④ 回归: 推荐(离线回退静态)/解释不走推荐/作文/写代码 ----------
c = "我想玩游戏，我有什么游戏推荐的吗？"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert xf._is_recommend(c), ("not recommend", c)
assert any(g in out for g in ("🔫", "🧭", "🎭", "🧩", "👥", "🎵")), ("rec no-emoji", out[:60])
_w("[ok] 游戏推荐(离线回退): ", out.strip().split("\n")[0][:36])

c = "推荐系统是什么"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert not xf._is_recommend(c) and not xf._is_essay(c), ("def misrouted", c, out[:30])
_w("[ok] 推荐系统是什么 走定义: ", out.strip().split("\n")[0][:36])

c = "帮我写一篇关于坚持的作文"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
body = out.split("\n\n", 1)[-1]
n = len(re.sub(r"\s", "", body))
assert xf._is_essay(c) and 400 <= n <= 900, ("essay routing/len", c, n)
_w("[ok] 作文 字数=", n)

c = "帮我写一个排序代码"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert not xf._is_essay(c), ("code->essay", out[:40])
_w("[ok] 写代码不走作文: ", out.strip().split("\n")[0][:24])

_w("SMOKE_OK cases=%d" % (len(online) + len(local) + len(id_cases) + 5))
os._exit(0)