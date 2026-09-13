# -*- coding: utf-8 -*-
"""
AI 小方 · FlphaLit Covi 1 —— 脚本 A：小说权重学习器
=========================================================================
作用
    把桌面《小方小说》里的 3 本 TXT（《方之旅途》《小方趣生活》《小方趣生活2》，
    约 110 万字）当成语料库，一句一句喂进**真正的** AI 小方：
    走 DeepThinkTransformer.train_step() 的真实反向传播 + 原生 AdamW 更新，
    一个劲地把权重调成"正常人说话"的样子（next-token 监督，和真实 AI 一致）。

    注意：只吃 .txt，绝不读同目录的 .pdf。（重装系统后语料在桌面）

关键事实（本机实测，Lite 档 = FlphaLit Wide3584 · 2.40B 就是 Lite 的参数量）
    实例化 ≈ 5-7 秒 / 常驻 ≈ 0.4-2.4GB（int8 权重是大头，这是 2.40B 本体的物理下限）；
    训练走 GPU（cupy，RTX 4070 Laptop）：窗口 192 首次一步 ≈ 15 秒（现场编译内核，只花一次），
    之后 ≈ 0.9 秒/步；
    可训练张量只有 6 个（gate + out_adapter + 末 2 层 FFN 低秩 A/B），
    大权重（int8 常驻）只前向、永不修改 —— 所以"参数量不变，权重真的在变"。

    ⚠ 别再被"卡死"骗了：v1.9.2 已根治 —— 真凶是 CuPy 把编译好的内核写缓存目录时被挡住
    （py-spy 抓到 cupy\cuda\_compiler_cache.py 的 _write_encoded）。引擎里已设
    CUPY_CACHE_IN_MEMORY=1，编译缓存只留内存、彻底绕开写盘。代价就是每个新进程首次
    要重编内核（约 15 秒），这是"绝不卡死"的买路钱，不是故障。

用法
    python 学习小说.py                  # 默认：Lite，窗口 192，最多学 2 遍（学到位就提前收工）
    python 学习小说.py --passes 1       # 只学 1 遍（赶时间）
    python 学习小说.py --hours 6        # 撑够 6 小时就收工落盘
    python 学习小说.py --steps 8        # 只跑 8 步（自检用，不写正式权重）
    python 学习小说.py --dry-run        # 自检模式：短跑 + 权重写到临时文件，不污染正式模型

断点续跑（合盖 / 断电 / 被杀，都不怕）
    每 100 步把「权重 + 断点」成对写一次（同一秒落盘，永远对得上）：
      · 权重 → xiaofang_train_covi1_<tier>.npz（引擎自带，按档位各一份）
      · 断点 → .xf_status\study_A_resume.json（学到第几遍 / 第几个文件 / 第几个窗口）
    下回直接 `python 学习小说.py` 就行 —— 它会自己读回权重、跳到上次那一窗接着学，
    不重复学、也不漏学。断点里带"语料清单 + 窗口 + 步长 + 遍数"指纹：任何一样变了
    就自动作废、老老实实从头学（绝不拿旧书的进度往新书上接）。
    合盖最多丢不到 100 步（约 1.5 分钟）的学习量。

学完自动落盘（xiaofang_train_covi1_<tier>.npz）并退出，不占后台。
内存闸门：软上限 5500MB（先回收再让路），硬上限 7000MB（落盘后立刻退出，绝不溢出）。
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import random
import signal
import sys
import time
import traceback

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # …\AI 小方
DESKTOP = os.path.dirname(HERE)                            # …\Desktop
STATUS_DIR = os.path.join(HERE, ".xf_status")

# 用户点名的那 3 本（存在就优先用；不存在则退化为"目录下所有 .txt"）
KNOWN_TXT = ("方之旅途.txt", "小方趣生活.txt", "小方趣生活2.txt")

CANDIDATE_DIRS = (
    os.path.join(DESKTOP, "小方小说"),
    os.path.join(HERE, "小方小说"),
    os.path.join(DESKTOP, "Desktop", "小方小说"),
)

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "big5")

DEFAULTS = dict(
    tier="lite",        # lite / pro / ultra（正式版三档闸门）
    window=192,         # 训练窗口（v1.9 提速：实测每 token 成本在 192 时最优；引擎侧由 XF_TRAIN_SEQ 放开）
    stride=192,         # 窗口滑动步长 → 不重叠，一遍过的步数最少（想更细可 --stride 96）
    passes=2,           # 最多学几遍（每遍都提前算 loss，明显不再降就收工）
    hours=0.0,          # 0 = 不限时；>0 则撑够这么多小时就落盘退出
    steps=0,            # 0 = 不限步数；>0 则只跑这么多步就落盘（自检用）
    lr_base=0.0018,     # 起手学习率（与引擎默认一致）
    lr_floor=0.0005,    # 余弦退火地板
    warmup=200,         # 前 200 步线性升温，避免一上来就炸 loss
    min_keep=0.65,      # 词表覆盖率低于这个值的窗口直接丢（别拿 id=0 垃圾喂模型）
    checkpoint=100,     # 每 N 步落盘一次（权重+断点成对写；一步约 1 秒，100 步≈100 秒，
                        #   就算合盖丢电也只丢不到两分钟的学习量）
    mem_soft=5500.0,    # 软内存上限 MB：先 gc，再让路（2.40B int8 本体 ≈2.4GB 是底线）
    mem_hard=7000.0,    # 硬内存上限 MB：落盘后立刻退出，绝不溢出
    log_every=20,       # 每 N 步打一行日志
    save_progress_every=5,   # 每 N 步写一次进度心跳 JSON
    nice=True,          # 降优先级，用户还要上课，别抢 CPU
    plateau=0.004,      # 一遍下来 loss 相对下降不足 0.4% → 认为学到位了
    quiet=True,         # 静音引擎自带的启动动画，日志只留我们自己的
)


def _pass_seed(p: int) -> int:
    """每一遍用不同（但可复现）的洗牌种子。"""
    return (20260913 + p * 7919) & 0x7FFFFFFF


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def _stdout_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        try:
            if s is not None and hasattr(s, "reconfigure"):
                s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def rss_mb() -> float:
    """当前进程常驻内存（MB）。拿不到就返回 -1，不影响主流程。"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        try:
            import ctypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            return -1.0


def lower_priority() -> bool:
    """把本进程降到"低于正常"优先级：小方在学，但用户上课不该被拖。"""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
        # I/O 也降一档，落盘时不跟用户抢磁盘
        if hasattr(psutil, "IONICE_LOW"):
            pass
        return True
    except Exception:
        return False


def hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return "{}h{}m{}s".format(h, m, s)
    if m:
        return "{}m{}s".format(m, s)
    return "{}s".format(s)


class Status:
    """把进度写成一个小 JSON（原子替换），串联器和用户都能随时看。"""

    def __init__(self, name: str):
        try:
            os.makedirs(STATUS_DIR, exist_ok=True)
        except Exception:
            pass
        self.path = os.path.join(STATUS_DIR, name + ".json")
        self._tmp = self.path + ".tmp"
        self.data = {"script": name, "pid": os.getpid(), "state": "starting",
                     "started_at": time.time(), "updated_at": time.time()}

    def set(self, **kw) -> None:
        self.data.update(kw)
        self.data["updated_at"] = time.time()
        self.data["heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self._tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(self._tmp, self.path)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 断点续跑：学到哪儿了记在这儿，合盖 / 断电 / 被杀都能接着来
# ----------------------------------------------------------------------------
RESUME_PATH = os.path.join(STATUS_DIR, "study_A_resume.json")
RESUME_VERSION = 3


def _resume_sig(window: int, stride: int, passes: int, files) -> str:
    """断点指纹：语料、窗口、步长、遍数任何一样变了，旧断点立刻作废。
    这样绝不会"拿着 A 语料的进度去 B 语料上接着学"，接错地方比从头再来糟得多。"""
    return json.dumps({"v": RESUME_VERSION, "w": int(window), "s": int(stride),
                       "p": int(passes), "f": [[f[0], int(f[2])] for f in files]},
                      ensure_ascii=False, sort_keys=True)


def load_resume():
    """读断点。文件不存在 / 坏了 / 版本不符 → 返回 None（一律当作从头开始）。"""
    try:
        with open(RESUME_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and int(d.get("version", -1)) == RESUME_VERSION:
            return d
    except Exception:
        pass
    return None


def save_resume(**kw) -> bool:
    """写断点。先写 .tmp 再 os.replace 原子替换，所以永远读不到半个文件。"""
    d = {"version": RESUME_VERSION, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    d.update(kw)
    try:
        os.makedirs(STATUS_DIR, exist_ok=True)
        tmp = RESUME_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, RESUME_PATH)
        return True
    except Exception:
        return False


def clear_resume() -> None:
    try:
        os.remove(RESUME_PATH)
    except Exception:
        pass


def resume_desc(d) -> str:
    """把断点说成人话。"""
    if not d:
        return "无"
    return "第 {} 遍 · 第 {} 个文件 · 该文件第 {} 个窗口（累计第 {} 步，引擎已学 {} 步）".format(
        d.get("pass", 1), int(d.get("file_idx", 0)) + 1, int(d.get("win_idx", 0)),
        d.get("step_all", 0), d.get("engine_steps", 0))



# ----------------------------------------------------------------------------
# 语料：定位 / 读取 / 切窗口
# ----------------------------------------------------------------------------
def find_corpus_dir(explicit: str = "") -> str:
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        raise SystemExit("[×] 你指定的语料目录不存在：{}".format(explicit))
    for d in CANDIDATE_DIRS:
        if os.path.isdir(d):
            return d
    # 最后的兜底：在桌面范围内找名为"小方小说"的目录
    try:
        for name in os.listdir(DESKTOP):
            p = os.path.join(DESKTOP, name)
            if os.path.isdir(p) and "小方小说" in name:
                return p
    except Exception:
        pass
    raise SystemExit("[×] 找不到《小方小说》文件夹（找过：{}）".format(" | ".join(CANDIDATE_DIRS)))


def _skip_name(n: str) -> bool:
    """`.` / `_` / `~$` 开头的一律不碰 —— 状态文件、断点、临时文件绝不能当语料读。"""
    return n.startswith(".") or n.startswith("_") or n.startswith("~$")


def list_txt(corpus_dir: str):
    """只收 .txt —— 同目录里那几个 .pdf / .docx 一概不要。**只往下钻一层。**

    · 顶层：`KNOWN_TXT` 点名的那三本优先，其余按名排序；
    · 子目录：脚本 B（爬取语料.py）的产出落在 `网络语料\\` 这种子文件夹里，
      这里跟着收一层（显示名写成 `网络语料/xxx.txt`，日志里好认）；
    · 一律跳过 `.` / `_` / `~$` 开头的文件与目录 —— 状态、断点、临时文件
      被当成语料喂进去是很危险的坑，宁可漏收也不误收。
    """
    entries = []                                   # (显示名, 绝对路径)
    try:
        for n in sorted(os.listdir(corpus_dir)):
            if _skip_name(n):
                continue
            p = os.path.join(corpus_dir, n)
            if os.path.isdir(p):
                try:
                    inner = sorted(os.listdir(p))
                except Exception:
                    continue
                for m in inner:
                    if _skip_name(m) or not m.lower().endswith(".txt"):
                        continue
                    q = os.path.join(p, m)
                    if os.path.isfile(q):
                        entries.append((n + "/" + m, q))
            elif n.lower().endswith(".txt"):
                entries.append((n, p))
    except Exception as e:
        raise SystemExit("[×] 读不了语料目录：{}".format(e))
    if not entries:
        raise SystemExit("[×] 目录里没有 .txt：{}".format(corpus_dir))

    by_name = {e[0]: e for e in entries}
    picked = [by_name[n] for n in KNOWN_TXT if n in by_name]      # 点名的三本优先
    rest = [e for e in entries if e not in picked and "/" not in e[0]]   # 顶层其它 .txt
    subdir = [e for e in entries if e not in picked and "/" in e[0]]     # 子目录里的语料

    files = []
    for name, p in picked + rest + subdir:
        try:
            sz = os.path.getsize(p)
        except Exception:
            continue
        if sz > 0:
            files.append((name, p, sz))
    if not files:
        raise SystemExit("[×] 语料全是空的？")
    return files


def read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ENCODINGS:
        try:
            txt = raw.decode(enc)
            if txt and "\ufffd" not in txt[:4000]:
                return txt
        except Exception:
            continue
    return raw.decode("utf-8", "ignore")


def paragraphs(text: str):
    """按行切段，顺手丢掉空行 / 纯符号行 / 太短的噪音行。"""
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if len(s) < 6:
            continue
        han = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
        if han < 4:                     # 至少 4 个汉字才算一句人话
            continue
        yield s


def scan_corpus(files, tok, terminator, window: int, stride: int, vocab=None):
    """先扫一遍统计 token / 窗口总量与词表覆盖率，好算进度与 ETA。
    不驻留语料，只数数。"""
    stat = []
    for name, path, size in files:
        text = read_text(path)
        chars = len(text)
        ntok = 0
        nhit = 0
        paras = 0
        for para in paragraphs(text):
            ts = tok.tokenize(para)
            ntok += len(ts) + (1 if terminator else 0)
            if vocab is not None:
                nhit += sum(1 for t in ts if t in vocab)
            paras += 1
        wins = max(0, (ntok - window) // max(1, stride) + 1) if ntok >= window else (1 if ntok >= 12 else 0)
        stat.append({"name": name, "chars": chars, "bytes": size, "paragraphs": paras,
                     "tokens": ntok, "windows": wins,
                     "coverage": (nhit / float(ntok)) if ntok else 0.0})
        del text
    return stat


def iter_windows(files, tok, terminator, window: int, stride: int):
    """懒加载地吐训练窗口：(第几个文件, 该文件第几个窗口, 文件名, token 列表)。

    前两个坐标是「断点续跑」用的门牌号：合盖中断后重跑，靠它俩精确跳回上次那一窗，
    既不重复学、也不漏学。顺序完全由 files 的次序决定，每次运行都一样，跳回才准。
    整本读完只留一个滑动缓冲，内存恒小。"""
    for fi, (name, path, _size) in enumerate(files):
        text = read_text(path)
        buf = []
        wi = 0
        for para in paragraphs(text):
            toks = tok.tokenize(para)
            if not toks:
                continue
            if terminator:
                toks = toks + [terminator]      # 段末打终止符：教它"一句话说到这儿为止"
            buf.extend(toks)
            while len(buf) >= window:
                yield fi, wi, name, buf[:window]
                wi += 1
                del buf[:stride]
        if len(buf) >= 12:                      # 书尾的零头也喂一口，别浪费
            yield fi, wi, name, buf[-min(len(buf), window):]
        del text, buf


# ----------------------------------------------------------------------------
# 学习率：线性升温 + 余弦退火
# ----------------------------------------------------------------------------
def make_lr_fn(base: float, floor: float, warmup: int, total: int):
    def fn(step: int) -> float:
        if step <= warmup:
            t = max(0.0, step / float(max(1, warmup)))
            return floor + (base - floor) * t
        p = (step - warmup) / float(max(1, total - warmup))
        p = min(1.0, max(0.0, p))
        return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * p))

    return fn


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="AI 小方 FlphaLit Covi 1 · 脚本 A：把 3 本小说喂成权重")
    ap.add_argument("--corpus", default="", help="语料目录（默认自动找桌面的《小方小说》）")
    ap.add_argument("--tier", default=DEFAULTS["tier"], choices=("lite", "pro", "ultra"))
    ap.add_argument("--passes", type=int, default=DEFAULTS["passes"], help="最多学几遍")
    ap.add_argument("--hours", type=float, default=DEFAULTS["hours"], help="撑够这么多小时就收工（0=不限）")
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"], help="只跑这么多步（0=不限）")
    ap.add_argument("--window", type=int, default=DEFAULTS["window"])
    ap.add_argument("--stride", type=int, default=DEFAULTS["stride"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr_base"])
    ap.add_argument("--checkpoint", type=int, default=DEFAULTS["checkpoint"], help="每 N 步落盘一次")
    ap.add_argument("--log-every", type=int, default=DEFAULTS["log_every"], help="每 N 步打一行日志")
    ap.add_argument("--min-keep", type=float, default=DEFAULTS["min_keep"],
                    help="窗口的词表命中率门槛，低于它整窗丢弃（默认 0.65）")
    ap.add_argument("--mem-hard", type=float, default=DEFAULTS["mem_hard"],
                    help="硬内存上限 MB，触到就落盘退出（默认 7000，2.40B 本体本身就要 ~2.4GB）")
    ap.add_argument("--mem-soft", type=float, default=DEFAULTS["mem_soft"],
                    help="软内存上限 MB，触到先 gc 再让路（默认 5500）")
    ap.add_argument("--dry-run", action="store_true",
                    help="自检：跑 --steps 步，权重写到 .xf_status 临时文件，不碰正式模型")
    ap.add_argument("--no-nice", action="store_true", help="不降进程优先级")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    global RESUME_PATH
    _stdout_utf8()
    args = parse_args(argv)

    tier = args.tier
    window = max(8, int(args.window))
    stride = max(1, min(int(args.stride), window))      # 允许 stride == window（不重叠）
    passes = max(1, int(args.passes))

    # CLI 覆盖配置表（下面统一引用 DEFAULTS，避免漏改漏算）
    DEFAULTS["log_every"] = max(1, int(args.log_every))
    DEFAULTS["min_keep"] = min(1.0, max(0.0, float(args.min_keep)))
    DEFAULTS["mem_hard"] = float(args.mem_hard)
    DEFAULTS["mem_soft"] = float(args.mem_soft)
    DEFAULTS["checkpoint"] = max(1, int(args.checkpoint))

    # ---- 引擎：档位闸门走正式版那套（XF_TIER），日志静音由我们自己做 ----
    os.environ["XF_TIER"] = tier
    # v1.9 提速关键：引擎的 train_step 默认把序列截到最后 TRAIN_MAX_SEQ(48) 个 token，
    #   光把本脚本的 --window 调大是**无效**的。这里显式把训练窗口闸门放开到我们想要的长度
    #   （只影响训练，聊天/生成那边一字不变）。必须在 import 引擎之前设好。
    os.environ["XF_TRAIN_SEQ"] = str(int(window))
    if DEFAULTS["quiet"] and not os.environ.get("XF_VERBOSE"):
        os.environ.setdefault("XF_QUIET", "1")
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    st = Status("study_A_dry" if args.dry_run else "study_A")
    if args.dry_run:
        RESUME_PATH = os.path.join(STATUS_DIR, "_dryrun_resume.json")   # 自检不碰正式断点
    t_start = time.time()
    st.set(state="boot", tier=tier)

    print("=" * 72)
    print(" AI 小方 · FlphaLit Covi 1 —— 脚本 A：小说权重学习器")
    print(" 档位={} · 窗口={} · 步长={} · 最多 {} 遍".format(tier.upper(), window, stride, passes))
    print("=" * 72, flush=True)

    if args.no_nice:
        pass
    elif lower_priority():
        print("[i] 已把本进程降到「低于正常」优先级，尽量不打扰上课。", flush=True)

    # ---- 语料 ----
    corpus_dir = find_corpus_dir(args.corpus)
    files = list_txt(corpus_dir)
    total_bytes = sum(f[2] for f in files)
    print("[1/4] 语料目录：{}".format(corpus_dir))
    for name, _p, size in files:
        print("      · {}  {:.2f} MB".format(name, size / 1048576.0))
    print("      合计 {:.2f} MB（只取 .txt，PDF/Word 一律不读）".format(total_bytes / 1048576.0), flush=True)
    st.set(state="corpus", corpus_dir=corpus_dir,
           files=[f[0] for f in files], corpus_mb=round(total_bytes / 1048576.0, 3))

    # ---- 建模型 ----
    print("[2/4] 正在构建引擎并载入已有权重…", flush=True)
    t0 = time.time()
    import xiaofang_covi1 as M          # 正式版主文件
    M.TRAIN_ENABLED = True
    xf = M.XiaoFang()
    tr = xf.transformer
    bank = tr.bank
    tk = xf.tokenizer
    if DEFAULTS["quiet"]:
        print("")
    print("      引擎就绪：{} · {:.3f}B 参数 · {} 层 / {} 头 · 用时 {:.1f}s · RSS {:.0f}MB".format(
        M.MODEL_TIER, M.MODEL_PARAMS / 1e9, M.MODEL_LAYERS, M.MODEL_HEADS,
        time.time() - t0, rss_mb()))
    print("      可训练张量 {} 个 / {} 个参数（fp32）；参与反向的层 {}".format(
        len(bank.params), bank.trainable_params(), tr.train_layers))
    print("      历史训练步数 {} · 权重文件 {}".format(bank.steps, tr._train_path()), flush=True)
    print("      训练窗口：脚本窗口 {} · 引擎实际生效闸门 {}（聊天/生成不受影响）".format(
        window, getattr(M, "_TRAIN_SEQ_CAP", window)), flush=True)

    if args.dry_run:
        tr._train_path = lambda: os.path.join(STATUS_DIR, "_dryrun_train.npz")
        print("[i] dry-run：权重只写临时文件，不动正式模型。", flush=True)

    # ---- 扫描 ----
    print("[3/4] 正在统计语料规模…", flush=True)
    t0 = time.time()
    stat = scan_corpus(files, tk, M.TERMINATOR, window, stride, vocab=set(tr.token2id.keys()))
    total_tokens = sum(s["tokens"] for s in stat)
    total_windows = max(1, sum(s["windows"] for s in stat))
    for s in stat:
        print("      · {:<16} {:>9,} 字 → {:>9,} token → {:>7,} 窗口 · 命中词表 {:.1%}".format(
            s["name"], s["chars"], s["tokens"], s["windows"], s["coverage"]))
    print("      合计 {:,.0f} token / {:,.0f} 窗口（一遍）；统计用时 {:.1f}s".format(
        total_tokens, total_windows, time.time() - t0), flush=True)

    # ---- 断点：上次学到哪儿了 ----
    #   指纹里含「语料清单 + 窗口 + 步长 + 遍数」，任意一样变了就作废重来 ——
    #   宁可从头学，也绝不拿旧书的进度去新书上接着跑（接错地方比从头再来糟得多）。
    sig = _resume_sig(window, stride, passes, files)
    resume = load_resume()
    if resume and resume.get("sig") != sig:
        print("[断点] 语料/窗口/步长/遍数跟上次不一样 → 旧断点作废，这次从头开始。", flush=True)
        clear_resume()
        resume = None
    if resume:
        print("[断点] 接着上次的进度学 ↓（权重已自动载回）", flush=True)
        print("      {}".format(resume_desc(resume)), flush=True)
    else:
        print("[断点] 没有可用断点，这次从头开始学。", flush=True)

    # ---- 训练 ----
    lr_fn = make_lr_fn(args.lr, DEFAULTS["lr_floor"], DEFAULTS["warmup"], total_windows)
    hard_steps = int(args.steps) if args.steps > 0 else 0
    deadline = (t_start + args.hours * 3600.0) if args.hours > 0 else 0.0

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
        print("\n[!] 收到中断信号({})，落盘后收工…".format(signum), flush=True)

    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        try:
            signal.signal(getattr(signal, sig), _on_signal)
        except Exception:
            pass

    print("[4/4] 开始学习（真实反向传播 + AdamW 调权重）…", flush=True)
    print("      注：本进程第一步要现场编译 GPU 内核，约 15 秒（只花这一次，之后每步约 1 秒），"
          "看到停住别慌，那是在编译，不是卡死。", flush=True)
    print("", flush=True)
    st.set(state="training", total_windows=total_windows, total_tokens=int(total_tokens),
           passes_planned=passes, lr=args.lr)

    # ---- 断点坐标：上次离开的那一窗 ----
    #   cursor = (第几遍, 第几个文件, 该文件第几个窗口)，指向"下一个该学的窗口"。
    #   循环里凡是排在 cursor 前面的窗口，一律跳过 —— 不重复学、不消耗步数。
    start_pass = max(1, min(int(resume.get("pass", 1)) if resume else 1, passes))
    cursor = (start_pass,
              int(resume.get("file_idx", 0)) if resume else 0,
              int(resume.get("win_idx", 0)) if resume else 0)

    step_all = int(resume.get("step_all", 0)) if resume else 0   # 累计步数：跨中断连续数，学习率才接得上
    run_steps = 0          # 本次运行真正走了多少步（--steps 按这个算）
    seen_windows = int(resume.get("seen_windows", 0)) if resume else 0
    kept = int(resume.get("kept", 0)) if resume else 0
    skipped = int(resume.get("skipped", 0)) if resume else 0
    keep_sum = 0.0
    keep_cnt = 0
    best_pass = None
    pass_log = []
    last_dump = time.time()
    recent = []            # 最近若干步 loss
    last_info = None

    all_done = False       # 是不是把所有遍数都学完了（学完才清断点）

    try:
        for p in range(start_pass, passes + 1):
            if stop["flag"] or (hard_steps and run_steps >= hard_steps):
                break
            order = list(files)
            if p > 1:
                random.Random(_pass_seed(p)).shuffle(order)   # 换个顺序再学一遍
            p_loss, p_n, p_t0 = 0.0, 0, time.time()
            print("—— 第 {}/{} 遍 · {} ——".format(p, passes, " ".join(f[0] for f in order)), flush=True)

            for fi, wi, name, toks in iter_windows(order, tk, M.TERMINATOR, window, stride):
                if stop["flag"]:
                    break
                if hard_steps and run_steps >= hard_steps:
                    break
                # ---- 断点续跑：排在 cursor 之前的窗口，上次已经学过了，跳过 ----
                if (p, fi, wi) < cursor:
                    continue
                if deadline and time.time() > deadline:
                    print("[i] 到点了（--hours），收工落盘。", flush=True)
                    stop["flag"] = True
                    break

                seen_windows += 1
                ids = [tr.token2id.get(t, 0) for t in toks]
                keep_rate = sum(1 for t in toks if t in tr.token2id) / float(len(toks))
                keep_sum += keep_rate
                keep_cnt += 1
                if len(set(ids)) < 3 or keep_rate < DEFAULTS["min_keep"]:
                    skipped += 1
                    continue

                step_all += 1
                run_steps += 1
                lr = lr_fn(step_all)
                if abs(bank.opt.lr - lr) > 1e-9:
                    bank.set_lr(lr)

                info = tr.train_step(ids[:-1], ids[1:])
                if not info:
                    skipped += 1
                    step_all -= 1
                    run_steps -= 1
                    continue
                kept += 1
                last_info = info
                loss = float(info["loss"])
                p_loss += loss
                p_n += 1
                recent.append(loss)
                if len(recent) > 200:
                    del recent[:len(recent) - 200]

                # 前 3 步逐步报一次：窗口放大到 192 后单步约 22 秒，若按 log_every(20) 算
                #   第一行日志要等 7 分钟以上，很容易被误判成"卡死"。先报头三步，之后恢复常态。
                #   断点续跑后同样先报头三步（run_steps），让人一眼看见"接上了、在动"。
                if run_steps <= 3 or kept <= 3 or kept % DEFAULTS["log_every"] == 0:
                    avg = sum(recent) / len(recent) if recent else 0.0
                    el = time.time() - t_start
                    if hard_steps:
                        eta = (hard_steps - run_steps) * (el / max(1, run_steps))
                    elif deadline:
                        eta = max(0.0, deadline - time.time())
                    else:
                        eta = max(0, total_windows - seen_windows) * (el / max(1, run_steps))
                    print("  步 {:>7,} │ loss {:6.3f} │ avg200 {:6.3f} │ gnorm {:6.2f} │ "
                          "lr {:.5f} │ 窗口 {:>7,}/{:<7,} │ RSS {:>5.0f}MB │ 已用 {} │ 剩 {}"
                          .format(step_all, loss, avg, float(info.get("gnorm", 0.0)), bank.opt.lr,
                                  seen_windows, total_windows, rss_mb(), hms(el), hms(eta)), flush=True)

                # ---- 落盘（权重 + 断点，成对写）----
                if kept % DEFAULTS["checkpoint"] == 0:
                    wrote = False
                    try:
                        wrote = bool(tr.dump_train_state())
                    except Exception:
                        traceback.print_exc()
                    # 权重写成功，才记"下一窗从哪开始"，两个文件同一秒落盘 → 续跑时永远对得上。
                    #   所以就算下一秒合盖断电，最多只丢这不到一个 checkpoint 的学习量。
                    if wrote:
                        save_resume(sig=sig, file_idx=fi, win_idx=wi + 1, step_all=step_all,
                                    kept=kept, skipped=skipped, seen_windows=seen_windows,
                                    engine_steps=int(bank.steps), **{"pass": int(p)})
                    last_dump = time.time()
                    print("  [存] 步 {} · 检查点已写入 {}{}".format(
                        step_all, tr._train_path(), "" if wrote else "（失败，下次接着来）"), flush=True)

                # ---- 进度 / 心跳 ----
                if kept % DEFAULTS["save_progress_every"] == 0:
                    st.set(state="training", pass_now=p, step=step_all, kept=kept,
                           skipped=skipped, seen_windows=seen_windows,
                           loss=round(loss, 4),
                           avg_loss=round(sum(recent) / len(recent), 4),
                           gnorm=round(float(info.get("gnorm", 0.0)), 4),
                           lr=round(bank.opt.lr, 6), rss_mb=round(rss_mb(), 1),
                           coverage=round(keep_sum / keep_cnt, 4) if keep_cnt else None,
                           elapsed=round(time.time() - t_start, 1),
                           engine_steps=int(bank.steps))

                # ---- 内存闸门 ----
                if kept % 10 == 0:
                    m = rss_mb()
                    if DEFAULTS["mem_hard"] > 0 and m > DEFAULTS["mem_hard"]:
                        print("[!] RSS {:.0f}MB 触到硬上限 {:.0f}MB → 落盘后立刻退出，绝不溢出。"
                              .format(m, DEFAULTS["mem_hard"]), flush=True)
                        stop["flag"] = True
                        break
                    if DEFAULTS["mem_soft"] > 0 and m > DEFAULTS["mem_soft"]:
                        gc.collect()
                        if rss_mb() > DEFAULTS["mem_soft"]:
                            time.sleep(0.5)      # 让路，别跟上课抢

            # ---- 每遍收尾 ----
            wrote_p = False
            try:
                wrote_p = bool(tr.dump_train_state())
            except Exception:
                traceback.print_exc()
            if wrote_p:
                # 一整遍学完了：断点挪到「下一遍 · 第 0 个文件 · 第 0 个窗口」
                save_resume(sig=sig, file_idx=0, win_idx=0, step_all=step_all, kept=kept,
                            skipped=skipped, seen_windows=seen_windows,
                            engine_steps=int(bank.steps), **{"pass": int(p) + 1})
            last_dump = time.time()
            mean_p = (p_loss / p_n) if p_n else float("nan")
            pass_log.append(round(mean_p, 4))
            print("  第 {} 遍完成 │ 本遍均 loss {:.4f} │ 已存 {} │ 累计步 {} │ 用时 {}"
                  .format(p, mean_p, tr._train_path(), step_all, hms(time.time() - p_t0)), flush=True)
            st.set(pass_done=p, pass_mean_loss=round(mean_p, 4), pass_log=pass_log,
                   step=step_all, kept=kept, skipped=skipped)

            # 学到位了就收工（第一遍不比）
            if best_pass is not None and best_pass > 0:
                rel = (best_pass - mean_p) / best_pass
                if rel < DEFAULTS["plateau"]:
                    print("  [✓] loss 相对只降了 {:.2%}（阈值 {:.2%}）→ 已经学到位，收工。"
                          .format(rel, DEFAULTS["plateau"]), flush=True)
                    clear_resume()      # 是主动判定"学到位了"，不是被打断 → 断点清零，下次从头
                    break
            best_pass = mean_p if best_pass is None else min(best_pass, mean_p)

            if stop["flag"] or (hard_steps and run_steps >= hard_steps):
                break
        else:
            all_done = True     # 所有遍数都学完了（没有中途 break）

    except KeyboardInterrupt:
        print("\n[!] 手动中断，落盘后收工…", flush=True)
    except MemoryError:
        print("\n[!] 内存告急，立刻落盘退出…", flush=True)
    except Exception:
        print("\n[×] 出错了，先把已有权重落盘：", flush=True)
        traceback.print_exc()

    # ---- 断点收尾 ----
    #   全部遍数学完 → 断点已无意义，清零；中途停下（合盖 / 断电 / 中断 / 步数到顶）→ 原样留着，下回接着来。
    if all_done:
        clear_resume()
        print("[断点] 全部遍数已学完 → 断点清零，下次运行从头开始。", flush=True)
    else:
        rd = load_resume()
        if rd:
            print("[断点] 已记下进度，下次运行会从这里接着学：{}".format(resume_desc(rd)), flush=True)

    # ---- 收尾 ----
    ok = False
    try:
        ok = bool(tr.dump_train_state())
    except Exception:
        ok = False
    el = time.time() - t_start
    avg_all = (sum(recent) / len(recent)) if recent else float("nan")
    print("\n" + "=" * 72)
    print(" 学习结束：跑 {} 步（有效 {} / 跳过 {}），用时 {}".format(step_all, kept, skipped, hms(el)))
    if keep_cnt:
        print(" 语料命中词表：{:.1%}（低于 {:.0%} 的窗口已丢弃，不用 id=0 的垃圾喂模型）"
              .format(keep_sum / keep_cnt, DEFAULTS["min_keep"]))
    print(" 引擎累计训练步数：{} │ 最近 200 步均 loss：{:.4f}".format(bank.steps, avg_all))
    if pass_log:
        print(" 每遍均 loss：{}".format(" → ".join(str(x) for x in pass_log)))
    if last_info:
        print(" 最后一步：loss {:.4f} │ gnorm {:.3f}".format(
            float(last_info["loss"]), float(last_info.get("gnorm", 0.0))))
    print(" 权重落盘：{} │ {}".format(tr._train_path(), "已写入 ✓" if ok else "写入失败 ×"))
    print(" 峰值内存：{:.0f} MB（硬上限 {:.0f} MB）".format(rss_mb(), DEFAULTS["mem_hard"]))
    print("=" * 72, flush=True)

    st.set(state="done", step=step_all, kept=kept, skipped=skipped,
           pass_log=pass_log, engine_steps=int(bank.steps),
           final_loss=round(avg_all, 4) if recent else None,
           elapsed=round(el, 1), rss_mb=round(rss_mb(), 1), saved=ok,
           finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
