# -*- coding: utf-8 -*-
"""AI 小方 v0.8 Pro - 冒烟测试. 静默(SHOW_DEEP_THINK=False), 跑完 os._exit(0) 强制释放内存."""
import io, sys, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import xiaofang_v08pro as XF
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"

ok = 0; n = 0
def check(name, cond, extra=""):
    global ok, n
    n += 1
    if cond:
        ok += 1
        print("  [PASS] " + name + (("  " + extra) if extra else ""))
    else:
        print("  [FAIL] " + name + (("  " + extra) if extra else ""))

t0 = time.time()
xf = XF.XiaoFang()
print("构建=%.1fs" % (time.time() - t0))

check("词库 >= 8000", xf.transformer.vocab_size >= 8000, "vocab=%d" % xf.transformer.vocab_size)
pc = xf.transformer.param_count()
sc = xf.transformer.storage_bytes()
check("参数量突破 10 亿", pc > 1e9, "param=%.3fB" % (pc / 1e9))
check("存储(含量化) <= 1GiB", sc < (1 << 30), "storage=%.3fGiB" % (sc / (1 << 30)))
check("旗舰 tier 生效", XF.MODEL_TIER.startswith("1.06B"), XF.MODEL_TIER)

def A(s):
    e = xf.emotion.analyze(s)
    i = xf.intent.detect(s, e)
    kb = xf.retriever.retrieve(s)
    return xf.reply(s, e, i, kb)

a = A("给我小方工作室的网址")
check("网址就只给网址", "fanggame.company" in a and "我是" not in a and "出品" not in a, "-> " + a)

a = A("你能干什么")
check("答能力非甩搜", ("代码" in a or "写代码" in a) and "去搜" not in a and "本地没存" not in a, "-> " + a)

a = A("引力波探测的原理是什么")
check("库外问题实体直指", "引力波" in a and "去搜" not in a and "本地没存" not in a, "-> " + a)

a = A("图灵完备怎么判定")
check("知识实体精确对齐直答", "图灵完备" in a and ("判断" in a or "判定" in a), "-> " + a)

print("冒烟: %d/%d 通过" % (ok, n))
os._exit(0)