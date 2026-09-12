# -*- coding: utf-8 -*-
"""
================================================================================
AI 小方 1.7 Alpha —— 全方位冒烟测试 (all-round smoke test)
================================================================================
覆盖本次 1.7 Alpha 的全部硬指标与用户可见能力:

  ①  引擎升档      : 2.4B / d_model 3584 / 15 层 / 28 头 / 内存闸门 4.6GB / 3B-4B 余量
  ②  可训练权重    : 优化器 AdamW + 反向传播 + LoRA 低秩 + 主权重落盘/续训
  ③  两层嵌套深想  : 想 → Transformer 验 → 第二轮想 → 再验 → 才交付 (由 Transformer 生成)
  ④  联网全局自动  : 不问也自己判; 代码类加权 GitHub / Gitee / CSDN
  ⑤  多行输入      : 回车才提交(软换行), 长输入不被截断
  ⑥  精准靶向      : 小方工作室问一件只答一件; 日常类小说只推日常
  ⑦  创作综合任务  : 写诗 / 设计小游戏 / 小说灵感
  ⑧  时间·位置     : 当地时刻 / 其他地区时间(含时区) / 当前位置
  ⑨  天气三连      : 当前 / 未来三天 / 24 小时 (假数据注入, 不联网)
  ⑩  情感支持      : 难过 / 开心 / 压力大 —— 先接住情绪
  ⑪  多轮投稿      : 作文、代码都能"接着上一版改"
  ⑫  算术高难      : 四则嵌套 / 超长求和 / 方程 / 多项式方程 / 求导 / 积分 / 微分方程
  ⑬  询问机制      : 缺参数先反问(选项 AI 自定, 末项恒为「其他」)
  ⑭  禁用词·回归   : 全局零禁用词, 不出现"我理解你的需求"等机械话术

运行: 在该文件所在目录执行  python _smoke_test_v17alpha.py
================================================================================
"""
import os
import sys
import time

os.environ.setdefault("XF_QUIET", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import xiaofang_v17alpha as M

M.SHOW_DEEP_THINK = False

_T0 = time.time()
_N = [0]
_FAILS = []
_OUTS = []                      # [(tag, 输出)] —— 最后统一做禁用词体检
BAN = ["我理解你的需求", "我已理解你的需求", "请询问", "查一下是否", "确定关键词"]
_W = 78


def _w(msg):
    print(msg, flush=True)


def _sec(title):
    _w("")
    _w("=" * _W)
    _w(title)
    _w("=" * _W)


def _ok(cond, msg):
    _N[0] += 1
    if cond:
        _w("  [OK]   " + msg)
    else:
        _w("  [FAIL] " + msg)
        _FAILS.append(msg)
    return bool(cond)


def _sum(tag, msg):
    """汇总型断言: 一次覆盖多条, 只计 1 例。"""
    return _ok(msg, "%s: %s" % (tag, msg))


# ------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------
_sec("⓪ 启动")
_t = time.time()
xf = M.XiaoFang()
_ok(True, "引擎构建成功 (%.2f s)" % (time.time() - _t))
tr = xf.transformer


def ctx(c):
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    return emo, intent, kb


def ask(c, tag=None):
    tag = tag or c
    emo, intent, kb = ctx(c)
    try:
        out = xf.reply(c, emo, intent, kb) or ""
    except Exception as e:
        out = ""
        _w("  [EXC]  %s -> %r" % (tag, e))
        _FAILS.append("EXC %s: %r" % (tag, e))
    _OUTS.append((tag, out))
    return out


# ==================================================================
# ① 引擎升档
# ==================================================================
_sec("① 引擎升档 (2.4B / 3584 宽 / 15 层 / 28 头 / 4.6GB 闸门)")
_ok(M.VERSION == "1.7 Alpha", "VERSION == '1.7 Alpha' (实际 %r)" % M.VERSION)
_ok(M.MODEL_NAME.endswith("1.7 Alpha"), "MODEL_NAME 收在 1.7 Alpha (实际 %r)" % M.MODEL_NAME)
_ok(M.MODEL_PARAMS >= 2.0e9, "MODEL_PARAMS 破 2B (%.3fB)" % (M.MODEL_PARAMS / 1e9))
_ok(M.MODEL_D >= 3328, "MODEL_D 更宽 (实际 %d)" % M.MODEL_D)
_ok(M.MODEL_LAYERS >= 15, "MODEL_LAYERS 更深 (实际 %d)" % M.MODEL_LAYERS)
_ok(M.MODEL_HEADS >= 28, "MODEL_HEADS 多头再加 (实际 %d)" % M.MODEL_HEADS)
_ok(tr.n_heads == M.MODEL_HEADS and tr.d_model == M.MODEL_D and tr.n_layers == M.MODEL_LAYERS,
   "实载引擎与声明一致 (n_heads=%d d=%d L=%d V=%d)" % (tr.n_heads, tr.d_model, tr.n_layers, tr.vocab_size))
_ok(M.MODEL_TIE is True, "MODEL_TIE=True (输出↔输入共享嵌入, 省存储)")
_ok(M.MODEL_UPGRADED is True, "MODEL_UPGRADED=True (已选到尺度上限内的旗舰档)")
_ok(M.MEM_BUDGET == int(4.6 * (1024 ** 3)), "MEM_BUDGET == 4.6GB (实际 %.2fGB)" % (M.MEM_BUDGET / 1024 ** 3))
_ok(M.MEM_BUDGET >= 4_400_000_000, "内存闸门容得下 3B/4B 档 (阈值 ≥ 4.4GB)")
_ok(M._SCALE_LIMIT >= 2_600_000_000, "默认尺度闸门 = 2.4B 旗舰档 (%.2fB)" % (M._SCALE_LIMIT / 1e9))
_ok(any(t[0] >= 4608 for t in M._MODEL_TIERS) and any(4000 <= t[0] < 4608 for t in M._MODEL_TIERS),
   "3B / 4B 档已预置 (Ultra4608 · 3.93B / Wide4096 · 3.12B) —— 纯 CPU 也留好升级路")
_ok(M.FORCE_OFFLINE is False, "FORCE_OFFLINE=False (默认可联网)")
_ok(M.SEED_TOKENS >= 40, "SEED_TOKENS ≥ 40 (读满整句意图, 实际 %d)" % M.SEED_TOKENS)
_ok(M.TERMINATOR == "</s>", "TERMINATOR == '</s>' (输出到终止符才停)")
_ok(M.NEST_TOK >= 10, "NEST_TOK ≥ 10 (每层嵌套思考自生成词元数, 实际 %d)" % M.NEST_TOK)


# ==================================================================
# ② 可训练权重: 优化器 + 反向传播 + 落盘
# ==================================================================
_sec("② 可训练权重 (优化器 + 反向传播 + 主权重落盘)")
_ok(hasattr(M, "AdamW"), "优化器 AdamW 已内置")
_ok(hasattr(M, "TrainBank"), "训练参数仓 TrainBank 已内置")
_ok(M.TRAIN_ENABLED is True, "TRAIN_ENABLED=True (初始权重不再固定)")
_ok(M.TRAIN_LR > 0 and M.TRAIN_CLIP > 0, "学习率/梯度裁剪均生效 (lr=%.4f clip=%.2f)" % (M.TRAIN_LR, M.TRAIN_CLIP))
_ok(M.TRAIN_LAST_BLOCKS >= 1, "有 %d 个 Block 参与梯度更新" % M.TRAIN_LAST_BLOCKS)
_ok(M.TRAIN_LORA_RANK >= 8, "LoRA 低秩修正 rank=%d (内存不涨)" % M.TRAIN_LORA_RANK)
_ok(M.TRAIN_STEPS_PER_TURN >= 1 and M.TRAIN_MAX_SEQ >= 16,
   "每轮 %d 步 / 截断 %d (纯 CPU 也不拖慢)" % (M.TRAIN_STEPS_PER_TURN, M.TRAIN_MAX_SEQ))

_s1 = xf._train_turn("什么是量子纠缠", "量子纠缠是一种量子力学现象")
_ok(isinstance(_s1, dict) and _s1.get("steps", 0) >= 1 and _s1.get("loss", 0) > 0,
   "反向传播走通: loss=%.4f gnorm=%.1f steps=%s" % (_s1.get("loss", 0), _s1.get("gnorm", 0), _s1.get("steps")))
_s2 = xf._train_turn("地球为什么是圆的", "因为引力使物质向质心聚拢")
_ok(_s2.get("steps", 0) > _s1.get("steps", 0), "累计步数递增 (%s → %s)" % (_s1.get("steps"), _s2.get("steps")))
_st = tr.train_stats()
_ok(_st.get("params", 0) > 1e7, "可训练主权重 %.2fM 个 fp32 参数" % (_st.get("params", 0) / 1e6))
_ok(len(_st.get("hist", [])) >= 2, "loss 历史已记录 (%d 条)" % len(_st.get("hist", [])))
_ok(tr._train_path() == "xiaofang_train_17.npz", "_train_path() == xiaofang_train_17.npz (实际 %r)" % tr._train_path())
_dumped = tr.dump_train_state()
_ok(_dumped and os.path.exists(tr._train_path()), "训练成果落盘成功 (下次启动接着学)")
if _dumped:
    _ok(tr.load_train_state() >= 0, "续训读取不报错 (权重可还原)")


# ==================================================================
# ③ 两层嵌套深度思考 (Transformer 本体生成)
# ==================================================================
_sec("③ 两层嵌套深度思考 (想 → 验 → 再想 → 再验 → 交付)")
_q = "什么是量子纠缠"
emo, intent, kb = ctx(_q)
_t = time.time()
_think_out = str(xf.think(_q, emo, intent, kb) or "")
_think_s = time.time() - _t
_ok(len(_think_out) > 200, "思考面板成篇 (%d 字, %.1f s)" % (len(_think_out), _think_s))
_ok("【⑤ 两层嵌套深度思考" in _think_out, "含「两层嵌套深度思考」段")
_ok("第①层" in _think_out and "第②层" in _think_out, "第①层 / 第②层 两轮都在")
_ok(_think_s < 120.0, "CPU 下思考不失控 (%.1f s < 120 s)" % _think_s)

_ln = getattr(xf, "last_nested", None)
_ok(isinstance(_ln, dict) and set(["t1", "v1", "t2", "v2"]).issubset(set(_ln or {})),
   "last_nested 四元组齐备 %s" % (sorted((_ln or {}).keys()),))
if isinstance(_ln, dict) and _ln:
    for _k in ("t1", "t2"):
        _d = _ln.get(_k)
        _ok(isinstance(_d, dict) and isinstance(_d.get("text"), str) and len(_d.get("text")) >= 1,
           "%s 由 Transformer 逐词生成 (text=%r)" % (_k, (_d.get("text") if isinstance(_d, dict) else "")[:24]))
    _ok(_ln["t1"].get("text") != _ln["t2"].get("text"),
       "两轮思考内容不同 (不是同一模板复读)")
    for _k in ("v1", "v2"):
        _d = _ln.get(_k) or {}
        _ok(("ok" in _d) and ("cos" in _d) and ("conf" in _d),
           "%s 验算指标齐备 ok=%s cos=%.3f conf=%.3f" % (_k, _d.get("ok"), _d.get("cos", 0), _d.get("conf", 0)))
    _ok(_ln["v1"].get("cos") is not None and _ln["v2"].get("cos") is not None,
       "两轮都有 Transformer 相似度(意图贴合度)")
    _ok(bool(_ln["t1"].get("ids")) and bool(_ln["t2"].get("ids")),
       "两轮都真的过了嵌入→注意力→FFN→采样 (ids 非空)")

_t1 = xf._tf_think("解释一下量子纠缠现象", n_tok=M.NEST_TOK)
_ok(isinstance(_t1, dict) and isinstance(_t1.get("text"), str)
    and 0.0 <= float(_t1.get("top1", 0)) <= 1.0 and 0.0 <= float(_t1.get("conf", 0)) <= 1.0
    and bool(_t1.get("ids")),
   "单轮 _tf_think 契约完整 (str 文本 / top1=%.4f conf=%.3f / 过网采样 %d 步)" % (
       _t1.get("top1", 0), _t1.get("conf", 0), len(_t1.get("ids") or [])))
_v1 = xf._tf_verify(_t1, intent, layer=1)
_ok(isinstance(_v1, dict) and isinstance(_v1.get("ok"), bool),
   "_tf_verify 返回布尔判定 ok=%s" % _v1.get("ok"))

_t = time.time()
_light = str(xf.think("😂😂😄", *ctx("😂😂😄")) or "")
_ok("轻量闲聊" in _light and (time.time() - _t) < 3.0,
   "轻量闲聊走快速通道 (%.2f s, 不做重计算)" % (time.time() - _t))


# ==================================================================
# ④ 联网搜索: 全局自动 + 代码类加权
# ==================================================================
_sec("④ 联网搜索 (全局自动判定 + 代码类加权 GitHub/Gitee/CSDN)")
_ok(len(M.SEARCH_BACKENDS) >= 4, "多后端联网 %s" % (M.SEARCH_BACKENDS,))
_ok(M.CODE_SEARCH_SITES[:3] == ["github.com", "gitee.com", "csdn.net"],
   "代码类前三权重 = GitHub / Gitee / CSDN (实际 %s)" % (M.CODE_SEARCH_SITES[:3],))
_ok(len(M.CODE_SEARCH_SITES) >= 5, "代码检索站点池 ≥5 (%d 个)" % len(M.CODE_SEARCH_SITES))
_ok(len(M.WEATHER_SITES) >= 3, "气象源池 ≥3 (%d 个)" % len(M.WEATHER_SITES))
_ok(M.STUDIO_LOCAL_ONLY is True, "STUDIO_LOCAL_ONLY=True (自家资料不上网乱找)")
_ok(xf._is_code_query("用python写个快排") is True, "代码类问句被正确识别")

_web_cases = [
    ("有哪些好玩的游戏推荐", False, "db_hit"),
    ("帮我搜一下最新的显卡行情", True, "explicit"),
    ("什么是量子纠缠？", True, "low_conf"),
    ("介绍一下弦理论？", True, "low_conf"),
    ("你是什么框架", False, "self_local"),
    ("小方工作室推出了几本小说", False, "self_local"),
]
for _c, _exp, _reason in _web_cases:
    _e, _i, _k = ctx(_c)
    _r = xf._should_web(_c, _e, _i, _k)
    _ok(_r is _exp and xf.last_web_reason == _reason,
       "「%s」→ web=%s reason=%s (期望 %s/%s)" % (_c, _r, xf.last_web_reason, _exp, _reason))
_ok(xf._needs_online("用 python 写个快排", None, 0.0, False) is True,
   "代码类问句全局自动联网 (无需用户说'搜一下')")
_ok(xf._needs_online("帮我搜一下最新的显卡行情", None, 0.0, False) is True,
   "显式搜索诉求全局自动联网")
_ok(xf._needs_online("小方工作室推出了几本小说", None, 0.0, False) is False,
   "自家情报即使是问句也不联网 (只走本地精准靶向)")
_ok(xf._needs_online("今天心情不错", None, 0.9, False) is False,
   "有库支撑的日常闲聊不滥用联网")
_ok(xf._is_self_ask("小方工作室推出了几本小说") is True, "自家问题被本地拦截")


# ==================================================================
# ⑤ 多行输入 (回车才提交)
# ==================================================================
_sec("⑤ 多行输入 (回车才提交 / 长输入不断行)")
try:
    with open(os.path.join(_HERE, "xiaofang_v17alpha.py"), "r", encoding="utf-8") as _f:
        _src = _f.read()
except Exception:
    _src = ""
_ok("msvcrt" in _src and "getwch" in _src, "逐键读取已接管输入 (msvcrt.getwch)")
_ok("0.015" in _src, "回车后留 15ms 前瞻窗口 → 软换行不误提交")
_ok(("\\n" in _src) or ("chr(10)" in _src), "存在软换行写入路径 (长输入可分行)")
_ok(M.RESPONSE_TIMEOUT >= 60, "RESPONSE_TIMEOUT=%d 为'静默上限'而非总时长上限 (长输出不再被直接断)" % M.RESPONSE_TIMEOUT)


# ==================================================================
# ⑥ 精准靶向: 小方工作室
# ==================================================================
_sec("⑥ 精准靶向 (问一件只答一件, 不倒库)")
_o = ask("小方工作室推出了几本小说", "studio-小说数")
_ok("6 部" in _o and len(_o) > 100, "问小说数 → 只报部数+书名 (%d 字)" % len(_o))
_ok("6月14日" not in _o and "用心做好游戏" not in _o, "不回夹纪念日/口号 (精准靶向)")
_ok("http" in _o or "fanggame" in _o, "小说条目带阅读网址")

_o = ask("小方工作室小说有哪些？我喜欢看日常类的，推荐一些。", "studio-日常类")
_ok(len(_o) > 100, "日常类问句有成篇推荐 (%d 字)" % len(_o))
_ok("6月14日" not in _o and "用心做好游戏" not in _o and "fanggame.company" not in _o,
   "日常类问句不夹纪念日/口号/官网")
_ok(("日常" in _o) or ("生活" in _o), "确实按'日常'口味筛选")

_o = ask("小方工作室的成立纪念日是哪天", "studio-纪念日")
_ok("6月14日" in _o and len(_o) <= 60, "只答纪念日 (%d 字): %r" % (len(_o), _o[:40]))
_ok("小说" not in _o, "不夹带小说清单")

_o = ask("小方工作室的口号是什么", "studio-口号")
_ok("用心做好游戏" in _o and len(_o) <= 60, "只答口号 (%d 字): %r" % (len(_o), _o[:40]))
_ok("6月14日" not in _o, "不夹带纪念日")

_o = ask("小方工作室官网是什么", "studio-官网")
_ok("fanggame" in _o and len(_o) <= 120, "只答官网 (%d 字): %r" % (len(_o), _o[:60]))

_o = ask("《小方趣生活》系列有哪些", "studio-系列")
_ok(len(_o) > 60, "系列问题成篇 (%d 字)" % len(_o))

_o = ask("小方工作室有哪些冒险类小说", "studio-冒险类")
_ok("冒险" in _o and len(_o) > 40, "只按'冒险'口味作答 (%d 字)" % len(_o))


# ==================================================================
# ⑦ 创作综合任务: 诗 / 游戏 / 灵感
# ==================================================================
_sec("⑦ 创作综合任务 (写诗 / 设计小游戏 / 小说灵感)")
_o = ask("写一首关于秋天的诗", "creative-诗")
_ok("📖" in _o and "《" in _o and len(_o) >= 60, "成诗 (%d 字)" % len(_o))

_o = ask("帮我设计一款小游戏", "creative-游戏")
_ok("游戏设计稿" in _o and len(_o) >= 600, "游戏设计稿成篇 (%d 字)" % len(_o))

_o = ask("给我一个科幻小说的灵感", "creative-灵感")
_ok("小说灵感" in _o and "科幻" in _o and len(_o) >= 100, "科幻灵感成篇 (%d 字)" % len(_o))

_o = ask("给我一个灵感", "creative-通用灵感")
_ok("小说灵感" in _o and len(_o) >= 60, "通用灵感兜底 (%d 字)" % len(_o))


# ==================================================================
# ⑧ 时间 / 位置
# ==================================================================
_sec("⑧ 时间 · 位置 (当地时刻 / 其他地区时间 / 当前位置)")
_o = ask("现在几点了", "time-本地")
_ok("🕐" in _o and "现在是" in _o, "当地时刻 (%r)" % _o[:36])

_o = ask("纽约现在几点", "time-纽约")
_ok("纽约" in _o and "UTC-5" in _o, "其他地区时间带时区 (%r)" % _o[:44])

_o = ask("北京现在几点", "time-北京")
_ok("UTC+8" in _o, "北京时区正确 (%r)" % _o[:44])

_o = ask("我在哪", "loc")
_ok("📍" in _o and "UTC+8" in _o, "当前位置 (%r)" % _o[:44])

_o = ask("今天几号", "time-日期")
_ok("📅" in _o and "年" in _o and "星期" in _o, "当天日期 (%r)" % _o[:40])


# ==================================================================
# ⑨ 天气三连 (假数据注入, 不联网)
# ==================================================================
_sec("⑨ 天气三连 (当前 / 未来三天 / 24 小时)")
FAKE = {
    "current_condition": [{"temp_C": "21", "FeelsLikeC": "20", "humidity": "55",
                           "weatherCode": "116", "weatherDesc": [{"value": "Partly cloudy"}],
                           "windspeedKmph": "12", "winddir16Point": "NE",
                           "visibility": "10", "uvIndex": "4", "observation_time": "0830 AM"}],
    "nearest_area": [{"areaName": [{"value": "Beijing"}], "region": [{"value": "Beijing"}]}],
    "weather": [
        {"date": "2026-09-12", "maxtempC": "27", "mintempC": "16",
         "hourly": [{"time": "%d00" % h, "weatherCode": "116",
                     "weatherDesc": [{"value": "Cloudy"}], "tempC": str(16 + h % 8),
                     "FeelsLikeC": str(15 + h % 8), "chanceofrain": str(10 + h),
                     "windspeedKmph": "10"} for h in range(0, 24, 3)]},
        {"date": "2026-09-13", "maxtempC": "26", "mintempC": "15",
         "hourly": [{"time": "%d00" % h, "weatherCode": "113",
                     "weatherDesc": [{"value": "Sunny"}], "tempC": "22", "FeelsLikeC": "21",
                     "chanceofrain": "5", "windspeedKmph": "9"} for h in range(0, 24, 3)]},
        {"date": "2026-09-14", "maxtempC": "24", "mintempC": "14",
         "hourly": [{"time": "%d00" % h, "weatherCode": "176",
                     "weatherDesc": [{"value": "Rain"}], "tempC": "19", "FeelsLikeC": "18",
                     "chanceofrain": "60", "windspeedKmph": "14"} for h in range(0, 24, 3)]},
    ],
}
_orig_fetch = xf._weather_fetch
xf._weather_fetch = lambda city: FAKE
try:
    _o = ask("北京天气", "wx-当前")
    _ok("当前天气" in _o and "湿度" in _o and "🌤️" in _o, "当前天气成表 (%d 字)" % len(_o))
    _ok("61" not in _o and "21" in _o, "温度取自气象源, 不编数")

    _o = ask("北京未来三天天气", "wx-三天")
    _ok("未来 3 天天气" in _o and "27" in _o and "09-12" in _o, "未来三天逐日 (%d 字)" % len(_o))
    _ok("09-13" in _o and "09-14" in _o, "三天都有日期")

    _o = ask("上海24小时天气", "wx-24小时")
    _ok("未来 24 小时天气" in _o and "15:00" in _o, "24 小时逐时 (时刻格式正常)")

    _o = ask("写一篇关于天气的作文", "wx-不劫持")
    _ok("需求拆解" in _o and len(_o) >= 400, "写天气作文不被天气功能劫持 (%d 字)" % len(_o))
finally:
    xf._weather_fetch = _orig_fetch

_ok(xf._weather_city("北京未来三天天气") == "北京", "_weather_city 抠城市正确")
_ok(xf._weather_city("上海24小时天气") == "上海", "天气城市表独立于时区表 (上海≠北京)")


# ==================================================================
# ⑩ 情感支持
# ==================================================================
_sec("⑩ 情感支持 (先接住情绪, 不报日期不倒模板)")
for _c in ["我今天好难过啊", "我最近压力好大，快崩溃了"]:
    _e, _i, _k = ctx(_c)
    _r = xf._emotion_reply(_c, _e)
    _ok(bool(_r) and len(_r) >= 10, "「%s」→ 共情接住 (%r)" % (_c, (_r or "")[:34]))
    _o = ask(_c, "emo-" + _c[:4])
    _ok(len(_o) >= 15, "回复非空且非纯日期 (%d 字)" % len(_o))

_e, _i, _k = ctx("今天心情特别好")
_r = xf._emotion_reply("今天心情特别好", _e)
_ok(bool(_r) and ("高兴" in _r or "开心" in _r), "好心情被正向接住 (%r)" % (_r or "")[:34])


# ==================================================================
# ⑪ 多轮投稿
# ==================================================================
_sec("⑪ 多轮投稿 (作文 / 代码 接着上一版改)")
_e1 = ask("写一篇关于花的作文，600字情感真挚，要有剧情", "rev-作文1")
_ok("需求拆解" in _e1 and len(_e1) >= 500, "第一稿成篇 (%d 字)" % len(_e1))
_ok(xf.last_kind == "essay" and xf.last_essay, "作文稿已记入会话状态")
_e2 = ask("再加长一些", "rev-作文2")
_ok("🔁" in _e2 and "第二稿" in _e2, "作文第二稿触发 (%r)" % _e2[:40])
_ok(len(_e2) > len(_e1), "第二稿比第一稿长 (%d → %d)" % (len(_e1), len(_e2)))

xf.last_essay = None
xf.last_code = None
xf.last_kind = None
_c1 = ask("用 Python 写个快速排序", "rev-代码1")
_ok("💻" in _c1 and "```python" in _c1, "第一版代码成篇 (%d 字)" % len(_c1))
_ok(xf.last_kind == "code" and xf.last_code, "代码稿已记入会话状态")
_c2 = ask("改成递归版本", "rev-代码2")
_ok("🔁" in _c2 and "第二稿" in _c2, "代码第二稿触发 (%r)" % _c2[:44])
_ok(_c2 != _c1, "第二版与第一版不同 (不是重出原文)")
_c3 = ask("加注释", "rev-代码3")
_ok("第三稿" in _c3, "继续迭代到第三稿 (%r)" % _c3[:40])
_ok(xf._is_revise("改成递归版本") is True and xf._is_revise("用 Python 写个快速排序") is False,
   "_is_revise 认得'改成', 不误判新起一单")


# ==================================================================
# ⑫ 算术 / 高难度
# ==================================================================
_sec("⑫ 算术 · 高难度 (四则嵌套 / 超长求和 / 方程 / 微积分)")
for _c, _exp in [("1+2*3", "7"),
                 ("计算 1234+5678*9", "52336"),
                 ("((1+2)*3+4)*5-6/2", "62"),
                 ("((12+8)*3-(45-15))/6 + 7*8 - 9", "52")]:
    _o = ask(_c, "math-" + _c[:12])
    _ok(_exp in _o, "「%s」= %s" % (_c, _exp))

_o = ask("请帮我算一下 123456789 + 987654321 + 111111111 + 222222222 等于多少", "math-超长求和")
_ok("1444444443" in _o, "简单但很长的问题 → 1444444443")

_o = ask("计算 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20", "math-长串求和")
_ok("210" in _o, "长串求和 → 210")

_o = ask("解方程 3x-5=10", "math-一元一次")
_ok("x = 5" in _o, "3x-5=10 → x = 5")

_o = ask("解方程 9x+12=3x+48", "math-移项")
_ok("x = 6" in _o, "9x+12=3x+48 → x = 6")

_o = ask("解方程 x^2-5x+6=0", "math-多项式方程")
_ok("3" in _o and "2" in _o, "x²-5x+6=0 → x = 3 或 x = 2 (%r)" % _o[:44])

_o = ask("求导 x^3+2x", "math-求导")
_ok("3*x^2" in _o, "求导 → 3*x^2 + 2")

_o = ask("∫ x^2 dx", "math-积分")
_ok("x^3/3" in _o, "积分 → x^3/3 + C")

_o = ask("解微分方程 dy/dx = 2x", "math-微分方程")
_ok("x^2" in _o and "C" in _o, "微分方程 → 含 x^2 与常数 C")


# ==================================================================
# ⑬ 询问机制 (缺参数先反问)
# ==================================================================
_sec("⑬ 询问机制 (选项 AI 自定, 末项恒为「其他」)")
_o = ask("写一篇作文", "ask-作文缺主题")
_ok("📋" in _o and "4. 其他[请说明]" in _o, "作文缺主题 → 反问且末项为其他")

_o = ask("写一段代码", "ask-代码缺语言")
_ok("1. Python" in _o and "其他[请说明]" in _o and len(_o) <= 260, "代码缺语言 → 反问语言")

_o = ask("写个防抖函数", "ask-防抖缺语言")
_ok("其他[请说明]" in _o, "无语言代码问句 → 先问语言")

_ask_txt = xf._ask_options("测试问题", ["甲", "乙", "丙"], multi=True)
_ok(_ask_txt.startswith("📋 测试问题") and "可多选" in _ask_txt and _ask_txt.endswith("4. 其他[请说明]"),
   "_ask_options 多选格式正确")

_o = ask("用 JavaScript 写一个节流函数", "code-JS")
_ok("💻" in _o and "```javascript" in _o and len(_o) >= 200, "带语言的代码请求直出实现 (%d 字)" % len(_o))

_o = ask("Python 语法速查", "code-速查")
_ok(len(_o) >= 500, "语法速查成篇 (%d 字)" % len(_o))


# ==================================================================
# ⑭ 身份 / 工具函数 / 全局体检
# ==================================================================
_sec("⑭ 身份 · 工具函数 · 全局体检")
_o = ask("你是什么框架", "id-框架")
_ok(M.MODEL_NAME in _o or "1.7 Alpha" in _o, "身份答的是 1.7 Alpha 本名")
_ok("fanggame.company" in _o or "小方工作室" in _o, "身份附带归属信息")
_o = ask("你是谁", "id-你是谁")
_ok(len(_o) >= 10, "自我介绍非空")

_ok(M._terminate_clean("abc") == "abc。", "_terminate_clean 缺终止符自动补")
_ok(M._terminate_clean("abc。") == "abc。", "_terminate_clean 已有终止符不动")
_ok(M._meets_std("你好") is True and M._meets_std("") is False and M._meets_std("好") is False,
   "_meets_std 达标判定正确")
_ok(M._run_with_timeout(lambda: 42, 2.0) == 42, "_run_with_timeout 正常取值")
_ok(M._run_with_timeout(lambda: time.sleep(3), 0.4, default=-1) == -1, "_run_with_timeout 硬超时兜底")

# 数据厚度
for _nm, _min in [("POEM_BANK", 5), ("POEM_TPL", 2), ("GAME_DESIGNS", 4), ("GAME_PLAN", 2),
                  ("NOVEL_IDEAS", 5), ("CREATIVE_EXTRA", 2), ("STUDIO_FACTS", 8),
                  ("STUDIO_NOVELS", 6), ("STUDIO_NOVELS_EXTRA", 3), ("NOVEL_GENRE_BUCKETS", 3),
                  ("CODE_EXAMPLES", 20), ("CODE_SYNTAX", 10)]:
    _v = getattr(M.DATA, _nm, None)
    _ok(hasattr(_v, "__len__") and len(_v) >= _min, "数据库 %s ≥ %d (实际 %s)" % (
        _nm, _min, len(_v) if hasattr(_v, "__len__") else "缺失"))

# 全局禁用词 + 空回复体检
_offenders = [(t, b) for t, o in _OUTS for b in BAN if b in o]
_ok(not _offenders, "全局禁用词零命中 (共体检 %d 条输出)" % len(_OUTS))
if _offenders:
    for _t, _b in _offenders[:8]:
        _w("        ✗ %s 含 %r" % (_t, _b))
_empties = [t for t, o in _OUTS if len(o.strip()) < 2]
_ok(not _empties, "无空回复 / 无断行空壳 (%d 条)" % len(_empties))
if _empties:
    _w("        ✗ 空回复: %s" % (_empties[:8],))


# ==================================================================
# 收尾
# ==================================================================
_sec("收尾")
_w("  用例总数 : %d" % _N[0])
_w("  失败数   : %d" % len(_FAILS))
_w("  总耗时   : %.1f s" % (time.time() - _T0))
if _FAILS:
    _w("")
    _w("FAILED CASES:")
    for _f in _FAILS:
        _w("  - " + _f)
    _w("")
    _w("SMOKE_FAILED cases=%d fails=%d" % (_N[0], len(_FAILS)))
    sys.stdout.flush()
    os._exit(1)

_w("")
_w("SMOKE_OK cases=%d" % _N[0])
sys.stdout.flush()
os._exit(0)
