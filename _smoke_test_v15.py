# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.5 正式版 —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0)。 在 v1.5 Alpha 基础上新增/强化:
#   ① Transformer 升档 ~1.68B 参数, 自适应档位确认
#   ② 高难数学: 三元一次方程组 + 矩阵加/乘/行列式
#   ③ 询问机制: 代码未确认语言 → 反问 "其他"末项; 多选选项构建
#   ④ Markdown 输出: 代码块 ``` / 粗体 ** / 分点 落地
#   ⑤ 靶向命中: 作文主题精确抽取实体(花), 不再反复 "这个话题"
import os
os.environ.setdefault("XF_QUIET", "1")
import sys, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v15 as M
M.SHOW_DEEP_THINK = False

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

_w("[i] 后端:", M._BACKEND, "| 版本:", M.VERSION)
assert M.VERSION == "1.5 正式版", ("version wrong", M.VERSION)
assert M.MODEL_NAME.endswith("1.5 正式版"), M.MODEL_NAME
xf = M.XiaoFang()
_w("[ok] 装配完成")
_tf = xf.transformer
assert hasattr(_tf, "intent_scorer"), "intent_scorer missing"

# ---------- ① Transformer 升档到 ~1.5-1.7B (维度再次增加) ----------
assert M.MODEL_PARAMS >= 1.4e9, ("params below 1.5B", M.MODEL_PARAMS)
assert M.MODEL_PARAMS <= 1.9e9, ("params above 1.7B-ish", M.MODEL_PARAMS)
_w("[ok] 档位:", M.MODEL_TIER, "| 参数:", round(M.MODEL_PARAMS / 1e9, 2), "B | d_model:", M.MODEL_D)
_w("[ok] 深度意图定向器深度:", _tf.intent_scorer.depth, "| d_model:", _tf.d_model); 
n = 5

def _ctx(c):
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    return emo, intent, kb, kb[0][0] if kb else 0.0

# ---------- ② 该联网/该本地 路由决策 (纯判断, 不联网) ----------
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

# ---------- ③ 问小方自己 → 身份本地答 ----------
for c in ["你是什么框架", "你的作者是谁", "你是谁"]:
    emo, intent, kb, _ = _ctx(c)
    out = xf.reply(c, emo, intent, kb) or ""
    assert out, "empty: " + c
    assert "没能联网" not in out, ("identity went to web", c, out[:40])
    assert ("小方" in out) or ("FlphaLit" in out) or ("工作室" in out), ("identity wrong", c, out[:40])
    _w("[ok] 身份本地答:", out.strip().split("\n")[0][:30]); n += 1

# ---------- ④ 深度意图定向 + 加权猜猜乐 ----------
import numpy as _np
q = "有什么好玩的游戏推荐"
seed = xf.tokenizer.tokenize(q)[:M.SEED_TOKENS] or ["好"]
ids = [_tf.token2id.get(t, 0) for t in seed] or [0]
vec = _tf.intent_encode(ids)
_vh = _np.asarray(M._to_host(vec))
assert _vh.shape == (_tf.d_model,), ("intent vec shape", _vh.shape)
assert float(_np.linalg.norm(_vh)) > 0, "intent vec zero"
_w("[ok] 意图向量 {0}维 范数={1}".format(_vh.shape[0], round(float(_np.linalg.norm(_vh)), 3))); n += 1

# ---------- ⑤ emoji 位置约束 ----------
assert "哈哈😂" not in M._normalize_emoji("哈哈😂你好啊"), ("mid-emoji kept", M._normalize_emoji("哈哈😂你好啊"))
_ = M._normalize_emoji("今天真开心，😂 你看这个")
assert "，😂" in M._normalize_emoji("今天真开心，😂 你看这个"), "punct-boundary emoji dropped"
assert M._normalize_emoji("真的好棒啊😂").rstrip().endswith("😂"), "tail emoji not kept"
_w("[ok] emoji 词中插→清走 / 逗号处→保留 / 句尾→保留"); n += 3

# ---------- ⑥ 靶向命中: 作文主题精确抽取实体(花), 不再反复 "这个话题" ----------
_c = "写一篇关于花的作文，600字情感真挚，要有剧情"
emo, intent, kb, _ = _ctx(_c)
assert xf._is_essay(_c), ("花作文未判作文", _c)
assert xf._extract_topic(_c) == ("花", "花"), ("主题未抽到「花」", xf._extract_topic(_c))
out = xf.reply(_c, emo, intent, kb) or ""
assert "花" in out, ("作文没出现花", out[:60])
assert "这个话题" not in out, ("还在重复『这个话题』", out[:60])
body = out.split("\n\n", 1)[-1]
_tlen = len(re.sub(r"\s", "", body))
assert 500 <= _tlen <= 800, ("作文字数异常", _tlen)
assert "\n\n" in out, ("作文未分段落(换行)", out[:60])
_w("[ok] 作文主题=「花」, 字数=", _tlen, "| 换行✓ | 语言✓ | 无'这个话题'"); n += 1

# ---------- ⑦ 代码到达率 + 询问机制 (语言未确认→反问, 末项恒为其他) ----------
_c = "写个冒泡排序给我"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "code", ("bubble not code", _c, intent["top"])
out = xf.reply(_c, emo, intent, kb) or ""
assert "请选择您的语言" in out, ("未反问语言", out[:60])
assert "其他[请说明]" in out, ("反问缺少『其他』", out[:60])
assert out.rstrip().endswith("[请说明]"), ("『其他』未作为末项", out[-30:])
assert "bubble_sort" not in out, ("语言未确认就硬写代码", out[:60])
_w("[ok] 冒泡未给语言→反问(末项其他): ", out.strip().splitlines()[0][:26]); n += 1

_c = "帮我用JS写个冒泡排序"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "code", ("js bubble not code", _c)
out = xf.reply(_c, emo, intent, kb) or ""
assert "JavaScript" in out and "bubbleSort" in out and "function" in out, ("no real JS bubble", out[:80])
assert "求和" not in out and "arraySum" not in out, ("js->sum bug!", out[:80])
assert "```" in out, ("代码未用 Markdown 代码块", out[:80])
_w("[ok] JS冒泡 → 真JS实现 + ```代码块(不再求和): ", out.strip().splitlines()[0][:26]); n += 1

_c = "C语言写个素数判断"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "code", ("c prime not code", _c)
out = xf.reply(_c, emo, intent, kb) or ""
assert "C 实现" in out and "isPrime" in out, ("no c prime", out[:80])
_w("[ok] C素数 → C 实现: ", out.strip().splitlines()[0][:26]); n += 1

_c = "帮我写一个排序代码"
emo, intent, kb, _ = _ctx(_c)
assert not xf._is_essay(_c), ("code->essay", out[:40])
_w("[ok] 写代码不让作文抢走"); n += 1

# ---------- ⑧ 循环教程(不再答创始人) + 纯表情轻量路径 ----------
_c = "写一个 Python 的基础的循环教程"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "code", ("not code", _c, intent["top"])
out = xf.reply(_c, emo, intent, kb) or ""
assert ("for" in out.lower() or "while" in out.lower()) and "循环" in out, ("no loop tutorial", out[:90])
assert "Guido" not in out and "创始人" not in out, ("still founder intro", out[:90])
_w("[ok] 循环教程→真实教程: ", out.strip().splitlines()[0][:26]); n += 1

_c = "😂😂😄"
emo2, int2, _kb2, _ = _ctx(_c)
_t0 = time.time()
r = xf.think(_c, emo2, int2, _kb2)
_dt = time.time() - _t0
assert "轻量闲聊" in str(r), ("light path not hit", _c)
assert _dt < 2.0, ("light path too slow", _dt)
_w("[ok] 纯表情轻量路径 {:.2f}s".format(_dt)); n += 1

# ---------- ⑨ 数学: 单变量方程 (回归) + 高难数学 ----------
_math_cases = [
    ("小方有道数学题我不会，x²=4，可以帮我解答一下吗？", "±2"),
    ("2x=6", "2"), ("x + 3 = 5", "2"), ("3x - 5 = 10", "5"),
]
for _c, _expect in _math_cases:
    emo, intent, kb, _ = _ctx(_c)
    assert intent["top"] == "math", ("not math", _c, intent["top"])
    out = xf.reply(_c, emo, intent, kb) or ""
    assert ("🧮" in out) and (_expect in out), ("math solve wrong", _c, out[:60])
    _w("[ok] 解方程 {} → 含{}:".format(_c, _expect), out.strip().splitlines()[0][:22]); n += 1

# 三元一次方程组 (v1.5 正式版)
_c = "解三元一次方程组: x+y+z=6, 2x-y+z=3, x+2y-z=2"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "math", ("三元非math", _c, intent["top"])
out = xf.reply(_c, emo, intent, kb) or ""
assert ("**x = 1**" in out) and ("**y = 2**" in out) and ("**z = 3**" in out), ("三元求解错", out[:80])
assert "元一次方程组" in out, ("三元无方程组字样", out[:60])
_w("[ok] 三元一次方程组 → x=1,y=2,z=3: ", out.strip().splitlines()[0][:24]); n += 1

_c = "解方程组: x+y=5, x-y=1"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "math", ("二元非math", _c, intent["top"])
out = xf.reply(_c, emo, intent, kb) or ""
assert ("**x = 3**" in out) and ("**y = 2**" in out), ("二元方程组错", out[:80])
_w("[ok] 二元方程组 → x=3,y=2"); n += 1

# 矩阵乘法 (含 Markdown 代码块)
_c = "计算矩阵 [[1,2],[3,4]] 和 [[5,6],[7,8]] 的乘积"
emo, intent, kb, _ = _ctx(_c)
assert intent["top"] == "math", ("矩阵非math", _c, intent["top"])
out = xf.reply(_c, emo, intent, kb) or ""
assert ("19, 22" in out) and ("43, 50" in out), ("矩阵乘法错", out[:80])
assert "```" in out, ("矩阵未用代码块", out[:60])
_w("[ok] 矩阵乘法 → [[19,22],[43,50]] + Markdown代码块"); n += 1

_c = "求 [[1,2],[3,4]] 的行列式"
emo, intent, kb, _ = _ctx(_c)
out = xf.math.answer(_c)
assert out and ("**-2**" in out), ("行列式错", out)
_w("[ok] 行列式 [[1,2],[3,4]] → det = -2"); n += 1

_c = "矩阵加法：[[1,2],[3,4]] 加 [[5,6],[7,8]]"
out = xf.math.answer(_c)
assert out and ("6, 8" in out) and ("10, 12" in out), ("矩阵加法错", out)
_w("[ok] 矩阵加法 → [[6,8],[10,12]]"); n += 1

# 反向: 解释/闲聊不被矩阵带偏
_c = "什么是推荐系统"
emo, intent, kb, _ = _ctx(_c)
out = xf.reply(_c, emo, intent, kb) or ""
assert "🧮" not in out, ("explain misrouted to math", out[:40])
_w("[ok] '什么是推荐系统' 仍走解释, 不误判数学"); n += 1

# ---------- ⑩ 数学 AST (函数/根号/常量/幂) ----------
_ast_cases = [("根号16等于多少", "4"), ("sqrt(81)是多少", "9"), ("sin30度等于多少", "0.5"),
              ("3的平方加4的平方等于多少", "25"), ("2加3乘以4等于多少", "14")]
for _c, _expect in _ast_cases:
    emo, intent, kb, _ = _ctx(_c)
    out = xf.reply(_c, emo, intent, kb) or ""
    assert ("🧮" in out) and (_expect in out), ("ast math failed", _c, out[:70])
    _w("[ok] 数学AST {} → 含{}:".format(_c, _expect), out.strip().splitlines()[0][:22]); n += 1

# ---------- ⑪ Markdown: 粗体 / 分点 落地 ----------
_ask = xf._ask_options("测试问题", ["甲", "乙", "丙"], multi=True)
assert "可多选" in _ask and "4. 其他[请说明]" in _ask, ("多选构建错", _ask)
assert _ask.rstrip().endswith("[请说明]"), ("末项非其他", _ask[-20:])
_w("[ok] 询问多选: 可多选 + 末项恒为其他: ", _ask.replace("\n", " / ")[:40]); n += 1

c = "3的平方加4的平方等于多少"
emo, intent, kb, _ = _ctx(c)
md = xf.reply(c, emo, intent, kb) or ""
assert "**" in md, ("未用 Markdown 粗体", md[:40])
_w("[ok] Markdown 粗体落地: ", md.strip().splitlines()[0][:24]); n += 1

# ---------- ⑫ 系统提示词 / 终止符 / 达标检测 就位 ----------
assert "1.5 正式版" in M.SYSTEM_PROMPT and "小方工作室" in M.SYSTEM_PROMPT, "SYSTEM_PROMPT identity missing"
assert "Markdown" in M.SYSTEM_PROMPT and "其他" in M.SYSTEM_PROMPT, "SYSTEM_PROMPT 询问/markdown 缺失"
assert M._terminate_clean("你好") == "你好。", "terminate_clean should append 。"
assert M._meets_std("还行") and not M._meets_std("  "), "meets_std boundary wrong"
_w("[ok] 系统提示词(身份/写码/计算/终止/达标) + Markdown + 询问 就位"); n += 1

_w("SMOKE_OK cases=%d" % n)
os._exit(0)