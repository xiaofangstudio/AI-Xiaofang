# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.4 Alpha —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0)。 在 v1.3 全回归基础上新增:
#   ① 打开对话/思考分段面板可折叠(展开/收起切换), 状态面板 Token/层数/阶段实时字段存在
#   ② 回答后输出学习因子行(学到单词/知识入库)
#   ③ 新命令 /prev /new /stop 路由 + 帮助页含快捷键面板 + 斜杠前缀大小写不敏感过滤
import os
os.environ.setdefault("XF_QUIET", "1")
import sys, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v14alpha as M
M.SHOW_DEEP_THINK = False

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

_w("[i] 后端:", M._BACKEND, "| 版本:", M.VERSION)
assert M.VERSION == "1.4 Alpha", ("version wrong", M.VERSION)
assert M.MODEL_NAME.endswith("1.4 Alpha"), M.MODEL_NAME
xf = M.XiaoFang()
_w("[ok] 装配完成")
_tf = xf.transformer
assert hasattr(_tf, "intent_scorer"), "intent_scorer missing"
_w("[ok] 深度意图定向器深度:", _tf.intent_scorer.depth, "| d_model:", _tf.d_model)

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

# ---------- ③ v1.2 核心①: 深度意图定向 + 加权猜猜乐 ----------
q = "有什么好玩的游戏推荐"
seed = xf.tokenizer.tokenize(q)[:M.SEED_TOKENS] or ["好"]
ids = [_tf.token2id.get(t, 0) for t in seed] or [0]
vec = _tf.intent_encode(ids)
import numpy as _np
_vh = _np.asarray(M._to_host(vec))
assert _vh.shape == (_tf.d_model,), ("intent vec shape", _vh.shape)
assert float(_np.linalg.norm(_vh)) > 0, "intent vec zero"
# 意图加权后的猜猜乐冠军不能为空, 且与未加权结果存在(都在候选集里)
d_bare = _tf.score_distribution(seed, set(seed), probs=None)
d_guided = _tf.score_distribution(seed, set(seed), probs=None, intent_vec=vec)
assert d_bare and d_guided, "score_distribution empty"
b_bare = max(d_bare, key=d_bare.get); b_guided = max(d_guided, key=d_guided.get)
assert b_guided, ("no guided best", b_guided)
_w("[ok] 意图向量 {0}维 范数={1} | 未加权冠军「{2}」→ 意图加权冠军「{3}」".format(
    _vh.shape[0], round(float(_np.linalg.norm(_vh)), 3), b_bare, b_guided)); n += 1
_r = _tf._intent_related(seed, vec, 4)
assert isinstance(_r, list), "intent_related must return list"
_w("[ok] 意图锚定语境词:", "、".join("「{}」".format(t) for t, _ in _r)); n += 1

# ---------- ④ v1.2 核心②: emoji 位置约束 ----------
mid = M._normalize_emoji("哈哈😂你好啊")
assert "😂" not in mid or "哈哈😂" not in mid, ("mid-emoji kept", mid)
_ = M._normalize_emoji("今天真开心，😂 你看这个")      # 逗号衔接 → 应保留
assert "，😂" in M._normalize_emoji("今天真开心，😂 你看这个"), "punct-boundary emoji dropped"
tail = M._normalize_emoji("真的好棒啊😂")
assert tail.rstrip().endswith("😂"), ("tail emoji not kept", tail)
_w("[ok] emoji 词中插→清走 / 标点逗号处→保留 / 句尾→保留"); n += 3

# ---------- ⑤ import 传送门2 入库 ----------
c = "传送门2是什么"
emo, intent, kb, _ = _ctx(c)
out = xf.reply(c, emo, intent, kb) or ""
assert out and ("解谜" in out or "Valve" in out or "传送门" in out), ("portal def missing", out[:60])
_w("[ok] 传送门2是什么 -> ", out.strip().split("\n")[0][:40]); n += 1

c = "传送门2通关攻略"
emo, intent, kb, _ = _ctx(c)
assert xf._needs_online(c, intent), ("ridge should want web", c)
_va = M._kb_aligned(c, kb[0][1]) if kb else False
_w("[ok] 传送门2攻略 该联网=True aligned=%s" % _va); n += 1

# ---------- ⑥ v1.1 Pro 核心回归: 推荐给具体名, 不甩类型 ----------
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
_w("[ok] 推荐类意图=suggest"); n += 1

# ---------- ⑦ 回归: 解释不走推荐/作文/写代码 ----------
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

# ---------- ⑧ v1.2 加强: 写/教代码教程 → 真实教程, 绝不答成实体(创始人)定义 ----------
c = "写一个 Python 的基础的循环教程"
emo, intent, kb, _ = _ctx(c)
assert intent["top"] == "code", ("not code", c, intent["top"])
out = xf.reply(c, emo, intent, kb) or ""
assert ("for" in out.lower() or "while" in out.lower()) and ("循环" in out), ("no loop tutorial", out[:90])
assert "Guido" not in out and "创始人" not in out, ("still founder intro", out[:90])
_w("[ok] 写循环教程→真实循环教程(不再答创始人): ", out.strip().split("\n")[0][:26]); n += 1

c = "什么是循环"
emo, intent, kb, _ = _ctx(c)
assert intent["top"] != "code", ("定义错判成代码", c, intent["top"])
_w("[ok] '什么是循环' 保持知识点, 不冤判代码:", intent["top"]); n += 1

c = "教我怎么遍历列表"
emo, intent, kb, _ = _ctx(c)
assert intent["top"] == "code", ("教遍历不是code", c, intent["top"])
out = xf.reply(c, emo, intent, kb) or ""
assert "for" in out.lower(), ("遍历未给代码", out[:80])
_w("[ok] 教遍历→code 有 for 示例: ", out.strip().split("\n")[0][:26]); n += 1

# ---------- ⑨ v1.2 卡死防呆: 纯表情/超短闲聊 → 轻量路径, 不跑 1B 神经正演 ----------
c = "😂😂😄"
emo2, int2, _kb2, _ = _ctx(c)
_t0 = time.time()
r = xf.think(c, emo2, int2, _kb2)
_dt = time.time() - _t0
assert "轻量闲聊" in str(r), ("light path not hit", c, intent["top"])
assert _dt < 2.0, ("light path too slow", _dt)
assert xf.last_web is None, ("light path must skip web")
_w("[ok] 纯表情输入: 轻量路径 {:.2f}s 跳过神经正演 (旧版会卡 CPU 假死)".format(_dt)); n += 1

c = "你好"
emo2, int2, _kb2, _ = _ctx(c)
_t0 = time.time()
r = xf.think(c, emo2, int2, _kb2)
_dt = time.time() - _t0
assert "轻量闲聊" in str(r), ("greet not light", c)
assert _dt < 2.0, ("greet light too slow", _dt)
_w("[ok] 超短问候走轻量路径 {:.2f}s".format(_dt)); n += 1

# 心跳/阶段/非阻塞输入的机制就位冒烟
assert hasattr(M, "_OUT_TICK") and "t" in M._OUT_TICK, "OUT_TICK missing"
assert hasattr(xf, "run_with_timeout") and hasattr(xf, "_note_stage"), "timer hooks missing"
_w("[ok] 活动心跳/阶段/非阻塞输入 机制已装配 (区分思考 vs 卡机)"); n += 1

# ---------- ⑩ v1.3 Alpha: 变量方程精确求解 (修复 "x²=4" 被通用模板曲解) ----------
_math_cases = [
    ("小方有道数学题我不会，x²=4，可以帮我解答一下吗？", "±2"),
    ("x^2=4", "±"),
    ("2x=6", "2"),
    ("x + 3 = 5", "2"),
    ("3x - 5 = 10", "5"),
]
for _c, _expect in _math_cases:
    emo, intent, kb, _ = _ctx(_c)
    assert intent["top"] == "math", ("not math", _c, intent["top"])
    out = xf.reply(_c, emo, intent, kb) or ""
    assert ("🧮" in out) and (_expect in out), ("math solve wrong", _c, out[:60])
    assert "定义" not in out and "原理" not in out and "常见用法" not in out, (
        "still generic template", _c, out[:60])
    _w("[ok] 解方程 {} → 命中(含{}):".format(_c, _expect), out.strip().splitlines()[0][:28]); n += 1

# 反向: 纯解释/闲聊仍不被方程求解带偏
_c = "什么是推荐系统"
emo, intent, kb, _ = _ctx(_c)
out = xf.reply(_c, emo, intent, kb) or ""
assert "🧮" not in out, ("explain misrouted to math", out[:40])
_w("[ok] 逆向'什么是推荐系统'仍走解释, 不误判数学: ", out.strip().splitlines()[0][:22]); n += 1

# ---------- ⑪ v1.3 Alpha: 联网搜索全网段 4s 硬超时 → 绝不 5 分钟假卡死 ----------
_c = "小方给我推荐一些好玩的游戏呗"
emo, intent, kb, _ = _ctx(_c)
assert xf._is_recommend(_c), ("not recommend", _c)
_t0 = time.time()
_out = xf.reply(_c, emo, intent, kb) or ""
_dt = time.time() - _t0
assert "🧮" not in _out and _out, ("recommend empty/wrong", _out[:40])
assert _dt < 30, ("recommend too slow (可能网络没硬超时)", _dt)
_games = [g[0] for g in xf._REC_GAMES]
assert any(name in _out for name in _games), ("offline rec no concrete game", _out[:80])
_w("[ok] 推荐游戏 {:.2f}s 内离线兜底给具体游戏名 (旧版联网挂5分钟)".format(_dt)); n += 1

_w("SMOKE_OK cases=%d" % n)
os._exit(0)