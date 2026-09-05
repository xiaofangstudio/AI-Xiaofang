# -*- coding: utf-8 -*-
"""AI 小方 v0.9 Alpha - 冒烟测试. 静默, 跑完 os._exit(0) 强制释放内存."""
import io, sys, os, time
os.environ["XF_QUIET"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import xiaofang_v09alpha as XF
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"

ok = n = 0
def check(name, cond, extra=""):
    global ok, n
    n += 1
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (("  " + extra) if extra else ""))
    if cond: ok += 1

t0 = time.time()
xf = XF.XiaoFang()
print("构建=%.1fs" % (time.time() - t0))

check("版本 0.9 Alpha", XF.VERSION == "0.9 Alpha", XF.VERSION)
check("词库 >= 8000", xf.transformer.vocab_size >= 8000, "vocab=%d" % xf.transformer.vocab_size)
pc = xf.transformer.param_count(); sc = xf.transformer.storage_bytes()
check("参数量突破 10 亿", pc > 1e9, "param=%.3fB" % (pc / 1e9))
check("存储(含量化) <= 1GiB", sc < (1 << 30), "storage=%.3fGiB" % (sc / (1 << 30)))

# 思考即时反馈 + 看门狗路径: 正常回答需能跑通且不卡
import io as _io, contextlib
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    xf.run_with_timeout("你好")
out = buf.getvalue()
check("思考中即时反馈弹出", "思考中" in out, "头部: " + (out.strip()[:16] or ""))

def A(s):
    e = xf.emotion.analyze(s); i = xf.intent.detect(s, e); kb = xf.retriever.retrieve(s)
    return xf.reply(s, e, i, kb)

a = A("给我小方工作室的网址")
check("网址就只给网址", "fanggame.company" in a and "我是" not in a and "出品" not in a, "-> " + a)

a = A("你能干什么")
check("答能力非甩搜", ("代码" in a or "写代码" in a) and "去搜" not in a and "本地没存" not in a, "-> " + a)

print("冒烟: %d/%d 通过" % (ok, n))
os._exit(0)