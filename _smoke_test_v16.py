# -*- coding: utf-8 -*-
# AI 小方 FlphaLit 1.6 正式版 —— 快速冒烟测试
# 静默 + 关闭深层思考 + 跑完 os._exit(0)。 在 v1.5 基础上新增/强化:
#   ① 引擎升档: ~2.4B 参数 / d_model 3584 / 15 层纵深, 常驻 int8 ≤ 3GB
#   ② 检索修复: 数据库优先 → 库里没有再联网 → 联网要进网页抠正文 → 再对齐整合
#   ③ 作文靶向: 主题 / 明确字数 / "必须包含"要点 三解析 + Markdown 排版 + 不重句
#   ④ 代码升级: 18 门语言语法表 / 代码示例库 / 完整可跑实现 / Markdown 代码块语言标注
#   ⑤ 数据扩充 + 系统提示词(检索纪律/需求拆解/何时询问/达标检测) 就位
import os
os.environ.setdefault("XF_QUIET", "1")
import sys, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xiaofang_v16 as M
M.SHOW_DEEP_THINK = False

def _w(*a):
    sys.stdout.write(" ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()

n = 0

def _ok(*a):
    # 统一计分: 一条断言通过就 n += 1
    global n
    n += 1
    _w("[ok]", *a)

_w("[i] 后端:", M._BACKEND, "| 版本:", M.VERSION)
assert M.VERSION == "1.6 正式版", ("version wrong", M.VERSION)
assert M.MODEL_NAME.endswith("1.6 正式版"), M.MODEL_NAME
assert M.FORCE_OFFLINE is False, ("默认不应强制离线", M.FORCE_OFFLINE)

xf = M.XiaoFang()
_w("[ok] 装配完成")
_tf = xf.transformer
assert hasattr(_tf, "intent_scorer"), "intent_scorer missing"

# ============================================================
# ① 引擎升档: ~2.4B 参数 / 3584 宽 / 15 层纵深 / 常驻 ≤ 3GB
# ============================================================
assert M.MODEL_PARAMS >= 2.0e9, ("参数量未破 2B", M.MODEL_PARAMS)
assert M.MODEL_PARAMS <= 2.9e9, ("参数量超出预期上限", M.MODEL_PARAMS)
assert M.MODEL_D >= 3328, ("d_model 未加宽", M.MODEL_D)
assert M.MODEL_LAYERS >= 13, ("纵深未加层", M.MODEL_LAYERS)
assert M.MODEL_FFN >= 4, ("FFN 倍率异常", M.MODEL_FFN)
assert M.MEM_BUDGET == 3 << 30, ("常驻内存硬约束不是 3GB", M.MEM_BUDGET)
assert M.MODEL_UPGRADED is True, ("未选到旗舰档位(内存不足被降级?)", M.MODEL_TIER)
_ok("档位:", M.MODEL_TIER, "| 参数:", round(M.MODEL_PARAMS / 1e9, 2), "B",
    "| d_model:", M.MODEL_D, "| 层数:", M.MODEL_LAYERS, "| 常驻约束:", M.MEM_BUDGET // (1 << 30), "GB")
_ok("深度意图定向器深度:", _tf.intent_scorer.depth, "| d_model:", _tf.d_model)

# ============================================================
# ② 检索修复: 数据库优先 → 库里没有再联网 → 进网页取正文 → 对齐整合
# ============================================================
def _ctx(c):
    emo = xf.emotion.analyze(c)
    intent = xf.intent.detect(c, emo)
    kb = xf.retriever.retrieve(c)
    return emo, intent, kb, (kb[0][0] if kb else 0.0)

# ②-1 数据库优先: 库里命中且实体对齐 → 就地取材, 不联网 (reason=db_hit)
_emo, _it, _kb, _ = _ctx("有哪些好玩的游戏推荐")
assert xf._should_web("有哪些好玩的游戏推荐", _emo, _it, _kb) is False, "库里已有具体条目却去联网"
assert xf.last_web_reason == "db_hit", ("不是走数据库优先", xf.last_web_reason)
_ok("数据库优先: 库里命中(score={}) → 不联网 reason={}".format(round(_kb[0][0], 2), xf.last_web_reason))

# ②-2 明确实时/让查 → 一定联网 (reason=explicit)
for _c in ["帮我搜一下最新的显卡行情", "现在几点了现在是几点搜一下", "查一下今天天气"]:
    _emo, _it, _kb, _ = _ctx(_c)
    assert xf._should_web(_c, _emo, _it, _kb) is True, ("明确实时诉求却不上网", _c)
    _ok("硬联网:", _c, "| reason:", xf.last_web_reason)

# ②-3 库里没有的概念题 → 自动上网 (reason=low_conf)  ← v1.6 修正的失效分支
for _c in ["什么是量子纠缠？", "介绍一下弦理论？", "什么是洛希极限"]:
    _emo, _it, _kb, _sc = _ctx(_c)
    assert _sc < 0.35, ("该用例本地命中偏高, 换用例", _c, _sc)
    assert xf._should_web(_c, _emo, _it, _kb) is True, ("库里没有概念题却不上网", _c)
    assert xf.last_web_reason == "low_conf", ("低置信未触发", _c, xf.last_web_reason)
    _ok("库里没有→上网:", _c, "| 本地score:", round(_sc, 2), "| reason:", xf.last_web_reason)

# ②-4 本地够用 / 自我身份 → 不上网 (reason=db_enough)
for _c in ["你是什么框架", "你的作者是谁", "你是谁"]:
    _emo, _it, _kb, _ = _ctx(_c)
    assert xf._should_web(_c, _emo, _it, _kb) is False, ("自我身份却联网", _c)
    assert xf.last_web_reason == "db_enough", ("未走本地够用", _c, xf.last_web_reason)
    _ok("本地够用:", _c, "| reason:", xf.last_web_reason)

# ②-5 联网后端顺序: 必应(直达链接, 能进网页) → 百度 → ddgs
assert hasattr(xf, "_bing_search"), "_bing_search 缺失(拿不到直达链接)"
assert hasattr(xf, "_baidu_search") and hasattr(xf, "_html_to_text"), "网页取词能力缺失"
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xiaofang_v16.py"),
            encoding="utf-8").read()
_p1, _p2, _p3 = _src.find("self._bing_search(query"), _src.find("self._baidu_search(query"), _src.find("for bk in SEARCH_BACKENDS")
assert 0 < _p1 < _p2 < _p3, ("联网后端顺序不是 必应→百度→ddgs", _p1, _p2, _p3)
_ok("联网后端顺序: 必应(直达链接) → 百度 → ddgs ✓")

# ②-6 进网页抠正文: 去脚本/样式/导航/页脚, 只留正文段落
_html = ("<html><head><style>.a{color:red}</style><script>var x=1;</script></head><body>"
         "<nav>首页 导航 登录 注册</nav>"
         "<article><h1>正文章节标题</h1><p>这是第一段正文内容，讲的是某个事物的来龙去脉。</p>"
         "<p>这是第二段正文，补充说明更多的细节信息。</p></article>"
         "<footer>版权所有 2026 某某公司</footer></body></html>")
_txt = xf._html_to_text(_html)
assert "第一段正文内容" in _txt and "第二段正文" in _txt, ("正文没抠出来", _txt[:80])
assert "var x=1" not in _txt, ("脚本没清掉", _txt[:80])
assert "版权所有" not in _txt, ("页脚没清掉", _txt[:80])
assert "正文章节标题" in _txt, ("h1 小标题被误丢", _txt[:80])
_ok("进网页抠正文: 留正文/h1 小标题, 去 script/style/nav/footer ✓")

# ②-7 _deep_read: 真进网页写回 item["page"], 拿不到就退回摘要
_orig_fetch = xf._fetch_page_text
xf._fetch_page_text = lambda href, max_chars=1400: ("这是从网页里抠出来的正文句子，内容足够长才能被采用，"
                                                    "用来验证确实进了网页而不是只看摘要。") * 3
_res = [{"title": "甲", "body": "摘要甲", "href": "http://example.com/a"},
        {"title": "乙", "body": "摘要乙", "href": "http://example.com/b"}]
_res = xf._deep_read(_res, want=2)
assert all(r.get("page") for r in _res), ("没写回 page", [r.get("page") for r in _res])
assert xf._deep_read([], want=2) == [], "空结果不该报错"
xf._fetch_page_text = lambda href, max_chars=1400: ""      # 取不到 → 退回摘要
_res2 = xf._deep_read([{"title": "丙", "body": "摘要丙", "href": "http://example.com/c"}], want=1)
assert "page" not in _res2[0], "取不到正文时不该硬塞 page"
xf._fetch_page_text = _orig_fetch
_ok("_deep_read: 进网页写回正文 / 取不到退回摘要(不卡死) ✓")

# ②-8 整合: 用"真抓到正文的那条"当出处, 且按问题对齐挑句、不无脑拼接
_emo, _it, _kb, _ = _ctx("游戏")
_pg = ("《黑神话：悟空》是国产动作游戏的代表作，画面与打击感都属上乘。"
       "无关的一句：天气预报说今天有雨记得带伞。"
       "《塞尔达传说》以开放世界自由探索著称，随便逛逛都有惊喜。")
xf.last_web = {"query": "好玩的游戏推荐", "ok": True, "time": time.time(),
               "results": [{"title": "游戏推荐榜", "href": "http://example.com/games",
                            "body": "短摘要", "page": _pg}]}
_out = xf.search_and_integrate("好玩的游戏推荐", _emo) or ""
assert "我进网页读过了" in _out, ("没走'进网页读过'这条", _out[:60])
assert "游戏推荐榜" in _out, ("出处用了摘要而非正文条目", _out[:60])
assert "http://example.com/games" in _out, ("没标来源", _out[:60])
assert "黑神话" in _out, ("正文内容没整合进来", _out[:80])
assert "，。" not in _out and "。。" not in _out, ("句子间糊成一坨/句读破损", _out[:80])
_ok("整合: 正文优先+相关挑句+标来源(进网页读过) ✓")
xf.last_web = None

# ============================================================
# ③ 作文靶向: 主题 / 明确字数 / "必须包含"要点 + Markdown 排版 + 不重句
# ============================================================
# ③-1 主题与字数解析
assert xf._is_essay("写一篇关于花的作文，600字情感真挚，要有剧情"), "作文未识别"
assert xf._extract_topic("写一篇关于花的作文，600字情感真挚，要有剧情")[0] == "花", "主题没抽到「花」"
assert xf._extract_topic("以坚持为题的作文，大约五百字")[0] == "坚持", "「以X为题」没抽到"
assert xf._extract_len("写一篇关于花的作文，600字") == 600, "阿拉伯数字字数没解析"
assert xf._extract_len("大约五百字") == 500, "中文数字字数没解析"
assert xf._extract_len("写一篇作文") is None, "没提字数却解析出了数"
assert not xf._is_essay("帮我写一个排序代码"), ("写代码被作文抢走", "排序代码")
_ok("作文解析: 主题=花/坚持, 字数=600/500/None, 写代码不抢 ✓")

# ③-2 「必须包含」要点解析: 多要点拆分 + 不把"800字"当要点
_must = xf._extract_must("写一篇作文，必须包含小方、星空、坚持，800字")
assert _must[:3] == ["小方", "星空", "坚持"], ("必须包含解析错", _must)
assert "800字" not in _must, ("字数被误当成要点", _must)
assert xf._extract_must("要有挫折要有成长") == ["挫折", "成长"], ("重复口令没拆开", xf._extract_must("要有挫折要有成长"))
_ok("必须包含解析:", _must, "| 重复口令拆分:", xf._extract_must("要有挫折要有成长"))

# ③-3 成文: 需求拆解小节 + Markdown 标题 + 要点全落 + 字数达标 + 段间空行
_c = "写一篇作文，必须包含小方、星空、坚持，800字"
_emo, _it, _kb, _ = _ctx(_c)
_out = xf.reply(_c, _emo, _it, _kb) or ""
assert "📋" in _out and "需求拆解" in _out, ("缺需求拆解小节", _out[:50])
assert "必须包含" in _out and "字数" in _out, ("拆解没写全要素", _out[:120])
assert "## " in _out, ("正文没用 Markdown 标题", _out[:120])
assert not _out.lstrip().startswith("📋 作文主题"), ("给了要点还反问主题", _out[:40])
for _mv in ("小方", "星空", "坚持"):
    assert _mv in _out, ("点名要点没落到文中", _mv)
_body = _out.split("\n\n## ", 1)[-1]
_tlen = len(re.sub(r"\s", "", _body))
assert 700 <= _tlen <= 1000, ("800字作文偏差过大", _tlen)
assert "\n\n" in _out, ("作文没分段(缺空行)", _out[:60])
_sents = [s.strip() for s in re.split(r"[。！？]", _body) if len(s.strip()) >= 8]
assert len(_sents) == len(set(_sents)), ("作文出现重复句子", len(_sents), len(set(_sents)))
assert "这个话题" not in _out, ("还在重复『这个话题』", _out[:60])
_ok("成文(800字+3要点): 字数=", _tlen, "| 需求拆解✓ | ## 正文✓ | 要点全落✓ | 不重句(",
    len(_sents), "句)✓")

# ③-4 按目标字数伸缩: 300 字不灌水, 500 字不含糊
for _c, _lo, _hi in [("写一篇关于花的作文，300字", 240, 420),
                     ("写一篇关于花的作文，500字", 420, 640)]:
    _emo, _it, _kb, _ = _ctx(_c)
    _o = xf.reply(_c, _emo, _it, _kb) or ""
    _tl = len(re.sub(r"\s", "", _o.split("\n\n## ", 1)[-1]))
    assert _lo <= _tl <= _hi, ("字数未按目标伸缩", _c, _tl)
    _ok("字数伸缩:", _c, "→", _tl, "字")

# ③-5 作文主题没给且没要点 → 反问(末项其他), 不硬写
_o = xf._maybe_ask("写一篇作文", {"top": "chat"})
assert _o and "其他[请说明]" in _o and _o.rstrip().endswith("[请说明]"), ("没反问主题", str(_o)[:60])
_ok("主题缺失→反问(末项恒为其他) ✓")

# ============================================================
# ④ 代码升级: 18 语言语法表 / 示例库 / 完整实现 / 语言标注不串
# ============================================================
assert len(M.DATA.CODE_SYNTAX) >= 18, ("语法表语言数不足", len(M.DATA.CODE_SYNTAX))
assert len(xf.code._BLOCK_TAG) >= 18, ("代码块语言标注不全", len(xf.code._BLOCK_TAG))
for _lg in ["Python", "JavaScript", "Java", "C", "C++", "C#", "Go", "TypeScript",
            "Kotlin", "Rust", "PHP", "Swift", "Ruby", "Lua", "R", "MATLAB", "SQL", "Shell"]:
    assert _lg in M.DATA.CODE_SYNTAX, ("缺语法表", _lg)
    assert _lg in xf.code._BLOCK_TAG, ("缺代码块标注", _lg)
_ok("18 门语言语法表 + 代码块标注齐备:", ", ".join(sorted(M.DATA.CODE_SYNTAX)[:6]), "…")

assert len(xf.code._all_tasks()) >= 60, ("任务/示例库偏少", len(xf.code._all_tasks()))
assert len(M.DATA.CODE_EXAMPLES) >= 30, ("代码示例库偏少", len(M.DATA.CODE_EXAMPLES))
_ok("代码任务+示例库:", len(xf.code._all_tasks()), "条 | CODE_EXAMPLES:", len(M.DATA.CODE_EXAMPLES), "条")

# ④-1 语言↔代码不错配: 命中语义里的语言就真给那门语言; 没有现成就标注"实际语言"
_t = {"kws": ["冒泡"], "title": "冒泡排序", "exp": "", "code": {"Python": "def bubble_sort(a): pass"}}
_lg2, _code2, _note2 = xf.code._pick(_t, "Rust")
assert _lg2 == "Python" and "bubble" in _code2, ("退而用现成版时语言标注错了", _lg2)
assert "Rust" in _note2 and "Python" in _note2, ("没说清现成是哪门语言", _note2)
_lg3, _c3, _n3 = xf.code._pick(_t, "Python")
assert _lg3 == "Python" and _n3 == "", ("指定语言命中时不该带备注", _lg3, _n3)
_ok("_pick 修正: 无现成语言→标注真实语言(Python)+说明, 不再张冠李戴 ✓")

# ④-2 未指定语言 → 反问, 绝不硬写
_emo, _it, _kb, _ = _ctx("写个冒泡排序给我")
assert _it["top"] == "code", ("冒泡没判代码", _it["top"])
_o = xf.reply("写个冒泡排序给我", _emo, _it, _kb) or ""
assert "请选择您的语言" in _o and "其他[请说明]" in _o, ("未反问语言", _o[:60])
assert _o.rstrip().endswith("[请说明]"), ("『其他』未作末项", _o[-24:])
assert "```" not in _o, ("语言未确认就硬写代码", _o[:60])
_ok("冒泡未给语言→反问(末项其他, 不硬写) ✓")

# ④-3 指定语言 → 真实现 + 代码块标注与语言一致
for _c, _tag, _sig in [("帮我用JS写个冒泡排序", "javascript", "bubbleSort"),
                       ("C语言写个素数判断", "c", "isPrime"),
                       ("用 Python 写个快速排序", "python", "quick_sort")]:
    _emo, _it, _kb, _ = _ctx(_c)
    _o = xf.reply(_c, _emo, _it, _kb) or ""
    assert "```" + _tag in _o, ("代码块语言标注不对", _c, _o[:70])
    assert _sig in _o, ("不是真实现", _c, _o[:70])
    assert "求和" not in _o and "arraySum" not in _o, ("答非所问(串到求和)", _c, _o[:70])
    _ok("真实现:", _c, "→ ```" + _tag, "+", _sig)

# ④-4 没有现成语言的请求 → 标注退回, 不冒充
_emo, _it, _kb, _ = _ctx("用 Rust 写个冒泡排序")
_o = xf.reply("用 Rust 写个冒泡排序", _emo, _it, _kb) or ""
assert "```python" in _o, ("退回版应标真实语言 python", _o[:70])
assert "Rust" in _o, ("应说明 Rust 原版可后补", _o[:70])
assert "Rust 完整实现" not in _o, ("标题谎称是 Rust 实现", _o[:70])
_ok("无现成语言: 标真实语言 + 说明可后补, 不谎称 ✓")

# ④-5 语法速查表
_emo, _it, _kb, _ = _ctx("Python 语法速查")
_o = xf.reply("Python 语法速查", _emo, _it, _kb) or ""
assert "语法速查" in _o and "```python" in _o, ("没喂语法表", _o[:60])
_ok("语法速查→真语法表 + 代码块 ✓")

# ④-6 循环教程不被"创始人"抢答
_emo, _it, _kb, _ = _ctx("写一个 Python 的基础的循环教程")
_o = xf.reply("写一个 Python 的基础的循环教程", _emo, _it, _kb) or ""
assert ("for" in _o.lower() or "while" in _o.lower()) and "循环" in _o, ("没给循环教程", _o[:80])
assert "Guido" not in _o and "创始人" not in _o, ("还在答创始人", _o[:80])
_ok("循环教程→真教程(非创始人) ✓")

# ============================================================
# ⑤ 数据文件扩充 + 系统提示词就位
# ============================================================
assert len(M.DATA.ESSAY_OPEN) >= 8 and len(M.DATA.ESSAY_BODY_A) >= 8, "作文句库偏少"
assert len(M.DATA.ESSAY_DEEP) >= 10 and len(M.DATA.ESSAY_CLOSE) >= 6, "作文句库偏少"
assert len(xf._MUST_TMPL) >= 5, ("必须包含要点模板偏少", len(xf._MUST_TMPL))
_ok("作文句库:", "lead/open/bodyA/B/C/fact/deep/close =",
    len(M.DATA.ESSAY_LEAD), len(M.DATA.ESSAY_OPEN), len(M.DATA.ESSAY_BODY_A),
    len(M.DATA.ESSAY_BODY_B), len(M.DATA.ESSAY_BODY_C), len(M.DATA.ESSAY_FACT),
    len(M.DATA.ESSAY_DEEP), len(M.DATA.ESSAY_CLOSE))

for _kw in ["1.6 正式版", "数据库优先", "Markdown", "需求拆解", "其他", "达标"]:
    assert _kw in M.SYSTEM_PROMPT, ("SYSTEM_PROMPT 缺: " + _kw)
_ok("系统提示词: 身份/数据库优先/需求拆解/Markdown/询问/达标 就位 ✓")

# ============================================================
# ⑥ 回归: 身份本地答 / 询问构建 / emoji 约束 / 终止符与达标检测
# ============================================================
for _c in ["你是什么框架", "你的作者是谁", "你是谁"]:
    _emo, _it, _kb, _ = _ctx(_c)
    _o = xf.reply(_c, _emo, _it, _kb) or ""
    assert _o, "empty: " + _c
    assert "没能联网" not in _o, ("身份跑去联网", _c)
    assert ("小方" in _o) or ("FlphaLit" in _o) or ("工作室" in _o), ("身份答错", _c)
_ok("身份本地答 ✓")

_ask = xf._ask_options("测试问题", ["甲", "乙", "丙"], multi=True)
assert "可多选" in _ask and "4. 其他[请说明]" in _ask, ("多选构建错", _ask)
assert _ask.rstrip().endswith("[请说明]"), ("末项非其他", _ask[-20:])
_ok("询问构建: 可多选 + 末项恒为其他 ✓")

assert "哈哈😂" not in M._normalize_emoji("哈哈😂你好啊"), "词中 emoji 没清走"
assert M._normalize_emoji("真的好棒啊😂").rstrip().endswith("😂"), "句尾 emoji 被丢"
_ok("emoji 词中插→清走 / 句尾→保留 ✓")

assert M._terminate_clean("你好") == "你好。", "terminate_clean 未补句号"
assert M._meets_std("还行") and not M._meets_std("  "), "meets_std 边界错"
_ok("终止符 / 达标检测 ✓")

# 纯表情轻量路径: 不该走重算力, 应在秒级返回
_emo, _it, _kb, _ = _ctx("😂😂😄")
_t0 = time.time()
_r = xf.think("😂😂😄", _emo, _it, _kb)
_dt = time.time() - _t0
assert "轻量闲聊" in str(_r), ("轻量路径没命中", str(_r)[:60])
assert _dt < 5.0, ("轻量路径过慢", _dt)
_ok("纯表情轻量路径 {:.2f}s ✓".format(_dt))

_w("SMOKE_OK cases=%d" % n)
os._exit(0)
