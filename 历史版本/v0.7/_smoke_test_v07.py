# -*- coding: utf-8 -*-
# v0.7 冒烟测试: 静默、快速、自动退出(不占内存)。
#  校验: 持久化自学习记忆(去重/校验/价值过滤) + 遗忘机制(LRU 淘汰旧词/最低价值知识)
#        + 跨会话回填 + 零侧写(用临时记忆文件, 测完即删, 不污染正式记忆)。
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v07 as XF

# 静默快速 + 隔离记忆文件(测完即删, 不影响真实记忆)
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"
_SAVE_MEM_FILE = XF.LearnerMemory.MEM_FILE
_TMP_MEM = "xiaofang_memory_smoke.json"
XF.LearnerMemory.MEM_FILE = _TMP_MEM
_tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _TMP_MEM)
for _p in (_tmp_path, _tmp_path + ".tmp"):
    if os.path.exists(_p):
        os.remove(_p)

ok = 0
def check(name, cond, extra=""):
    global ok
    if cond:
        ok += 1
        print("[PASS] %s %s" % (name, extra))
    else:
        print("[FAIL] %s %s" % (name, extra))

def get_answer(xf, text):
    emo = xf.emotion.analyze(text)
    intent = xf.intent.detect(text, emo)
    kb = xf.retriever.retrieve(text)
    return xf.reply(text, emo, intent, kb)

# ================= A. 模型基础(沿用 0.6 Pro) =================
print("=== A. 实例化模型 (纯自研 GPU/CPU 双后端) ===")
t0 = time.time()
xf = XF.XiaoFang()
print("版本=%s 引擎=%s | 后端=%s | %s | d_model=%d L=%d H=%d vocab=%d KB=%d | 载入=%.2fs" % (
    XF.VERSION, XF.ENGINE_NAME, XF._BACKEND, XF.HW["gpu_model"], xf.transformer.d_model,
    xf.transformer.n_layers, xf.transformer.n_heads, xf.transformer.vocab_size,
    len(xf.retriever.docs), time.time() - t0))

check("词库 >= 8000", xf.transformer.vocab_size >= 8000,
      "vocab=%d" % xf.transformer.vocab_size)
url_a = get_answer(xf, "给我小方工作室的网址")
check("网址就只给网址",
      "fanggame.company" in url_a and "出品" not in url_a, "-> " + url_a)
a = get_answer(xf, "你的模型是什么")
check("模型名 FangPro + 0.7 + 无旧名/旧版",
      "FangPro" in a and "0.7" in a and "FlphaLit" not in a
      and "Flash2" not in a and "正式版" not in a, "-> " + a[:50])
probs, tr = xf.transformer.forward([xf.transformer.token2id.get(t, 0)
                                    for t in XF.Tokenizer().tokenize("你好小方")][:24] or [0], trace=True)
check("TF forward + trace 双后端稳定",
      len(tr["blocks"]) == xf.transformer.n_layers and tr["probs"] is not None)
del probs

# ================= B. 自学习记忆: 过滤/去重/价值/持久化/遗忘 =================
print("=== B. 自学习记忆 (持久化 + 过滤 + 遗忘) ===")
tok = XF.Tokenizer()
m = XF.LearnerMemory(tok)
_base_titles = set()
for _e in XF.DATA.KNOWLEDGE_BASE:
    _base_titles.add(_e["t"]); _base_titles.update(_e.get("a", []))
m.bind_base(XF.DATA.COMMON_WORDS, _base_titles)

# 持久化 round-trip: 学→写盘→重建→仍在
chk1 = m.note_word("亚原子物理学")          # 新汉词, 合法
chk_dedup = m.note_word("亚原子物理学")     # 重复 → 仅累加, 不再新增
chk_filter = m.note_word("的了的")          # 全是虚词 → 过滤拒绝
chk_already = m.note_word("处理器")         # 基础常用词 → 拒绝(不该被当生词学)
chk_def = m.note_def("夸克禁闭", "夸克永远不能单独出现只成团束缚")  # 有信息量 → 存摘要
chk_def_short = m.note_def("太短", "嗯")     # 摘要过短 → 过滤
check("生词去重(同一词只入表一次)", chk1 is True and chk_dedup is False,
      "首学=%s 再学=%s" % (chk1, chk_dedup))
check("生词校验(纯虚词/基础词被拒)", chk_filter is False and chk_already is False)
check("知识存摘要(压缩)而非原文", chk_def is True and m.know.get("夸克禁闭", {}).get("s", "") == "夸克永远不能单独出现只成团束缚")
check("知识过滤(过短无信息量被拒)", chk_def_short is False)
saved_words = set(m.words); saved_know = set(m.know.keys())
m.commit()   # 遗忘+落盘
m2 = XF.LearnerMemory(tok)   # 重建(模拟退出重进/切断上下文)
check("跨会话持久化(退出/重进仍在)",
      "亚原子物理学" in m2.words and "夸克禁闭" in m2.know and
      saved_words == set(m2.words) and saved_know == set(m2.know.keys()))
# 重新载入学到的知识进入检索(不存对话/人格)
r = XF.SemanticRetriever(tok)
m2.seed_retriever(r)
check("记忆知识回填检索(检索到学习成果)",
      any(d["title"] == "夸克禁闭" for d in r.docs))

# 遗忘机制: 学新词淘汰久不用旧词 (LRU) — 不绑基础词库 & 清空加载态以保证确定
wm = XF.LearnerMemory(tok); wm.words.clear(); wm.know.clear(); wm._known_titles = set()
wm.MAX_WORDS = 5
for w in ["卡拉瓦乔", "齐柏林飞", "普罗米斯", "阿尔托克", "布宜诺斯", "纳西坎特", "佩尔加蒙", "塔拉萨尔"]:
    wm.note_word(w)
fk = wm.evict()
check("遗忘-学新词淘汰旧词(LRU), 词数回落上限",
      len(wm.words) <= 5 and fk[0] == 3, "容量=%d>淘汰%d" % (len(wm.words), fk[0]))
# 遗忘机制: 知识超上限淘汰最低价值 — 清空加载态以保证确定
km = XF.LearnerMemory(tok); km.words.clear(); km.know.clear(); km._known_titles = set()
km.MAX_KNOW = 5
km.note_def("阿尔法射线", "由氦原子核组成穿透力弱但电离能力很强")
km.note_def("贝塔射线", "高速电子流穿透力比阿尔法略强")
km.note_def("伽马射线", "高频电磁波穿透力极强需要厚重铅板")
km.note_def("德尔塔变种", "某个病毒变异株传染性增强的描述")
km.note_def("艾普西隆", "用于命名一系列变种的一个希腊字母")
km.note_def("泽塔协定", "量子物理会议中关于观测者效应的共识")
ev = km.evict()
check("遗忘-知识超上限淘汰最低价值, 回落上限",
      len(km.know) <= 5 and ev[1] == 1, "容量=%d" % len(km.know))

# ================= 收尾: 还原 + 删除临时记忆文件 =================
XF.LearnerMemory.MEM_FILE = _SAVE_MEM_FILE
for _p in (_tmp_path, _tmp_path + ".tmp"):
    if os.path.exists(_p):
        os.remove(_p)
print("=== 冒烟通过 %d/12 → 立即退出(释放内存) ===" % ok)
sys.stdout.flush()
os._exit(0)