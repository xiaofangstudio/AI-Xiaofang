# -*- coding: utf-8 -*-
"""
AI 小方 · FlphaLit Covi 1 —— 脚本 B：网络语料权重学习器（前台可见版）
=========================================================================
它是什么
    把桌面上爬回来的《网络语料》TXT（默认目录：<桌面>\\小方小说\\网络语料\\语料_*.txt）
    当成第二份语料库，一句一句喂进**真正的** AI 小方：
    走 DeepThinkTransformer.train_step() 的真实反向传播 + 原生 AdamW，
    把权重继续往"正常人说话"的方向调（next-token 监督，和真实 AI 一致）。

    和脚本 A（学习小说.py）的关系：
      · 学的是同一个模型本体、写的是同一个权重文件，所以两份语料的知识会累加；
      · 断点各记各的（脚本 A 用 study_A_resume.json，脚本 B 用 study_B_resume.json），
        互不覆盖、互不干扰，两个脚本也可以先后跑；
      · 只吃 .txt。爬取脚本的状态目录（.crawl_status / .xf_status 这类下划线或点开头的）
        一律不会被当语料读进来。

前台，不后台（用户要求）
    · 本脚本就是一个普通的前台进程：你双击 .bat 或 python 跑它，命令行窗口就一直在，
      所有进度都打在**这个窗口**里，不 fork、不 detach、不留后台；
    · 进度是一条不停刷新的实况行：
        [####......] 62.3%  窗口 12,345/19,800  步 12,345  loss 3.881
        已用 1h02m03s  剩 0h37m20s  RSS 2630MB
      "剩"是实时倒推的剩余时间，做得快它就掉得快；
    · 每隔 log_every 步，同一条实况行会作为**永久日志**换行打一遍，方便回看。

用法
    python 学习语料.py                  # 默认：Lite 档，窗口 192，最多学 2 遍
    python 学习语料.py --tier pro       # 用 Pro 档学（权重写 xiaofang_train_covi1_pro.npz）
    python 学习语料.py --passes 1       # 只学 1 遍
    python 学习语料.py --hours 3        # 撑够 3 小时就落盘收工
    python 学习语料.py --steps 8        # 只跑 8 步（自检）
    python 学习语料.py --dry-run        # 自检：权重写临时文件，不污染正式模型
    python 学习语料.py --scan-only      # 只统计语料规模，不进训练

断点续跑（合盖 / 断电 / 被杀都不怕）
    每 100 步把「权重 + 断点」成对写一次（同一秒落盘，永远对得上）：
      · 权重 → xiaofang_train_covi1_<tier>.npz（引擎自带）
      · 断点 → .xf_status\\study_B_resume.json（学到第几遍 / 第几个文件 / 第几个窗口）
    下回直接再跑一次就行，它会读回权重、跳到上次那一窗接着学，不重复、不漏学。
    断点里带「语料清单 + 窗口 + 步长 + 遍数」指纹：任意一样变了就自动作废、从头学。
    合盖最多丢不到 100 步。

内存闸门
    软上限 5500MB（先 gc 再让路），硬上限 7000MB（落盘后立刻退出，绝不溢出）。
    2.40B 本体的 int8 权重常驻本身就是 ~2.4GB，这是物理下限，压不下去。
"""

from __future__ import annotations

import argparse
import gc
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

# 爬取脚本的产出目录（脚本 B 的默认语料）：<桌面>\小方小说\网络语料\
CORPUS_SUBDIR = "网络语料"
CORPUS_PARENT = "小方小说"

CANDIDATE_CORPUS_DIRS = (
    os.path.join(DESKTOP, CORPUS_PARENT, CORPUS_SUBDIR),
    os.path.join(HERE, CORPUS_PARENT, CORPUS_SUBDIR),
    os.path.join(DESKTOP, CORPUS_SUBDIR),
    os.path.join(HERE, CORPUS_SUBDIR),
)

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "big5")

DEFAULTS = dict(
    tier="lite",        # lite / pro / ultra（正式版三档闸门）
    window=192,         # 训练窗口（引擎侧由 XF_TRAIN_SEQ 放开）
    stride=192,         # 窗口滑动步长 → 不重叠
    passes=2,           # 最多学几遍
    hours=0.0,          # 0 = 不限时
    steps=0,            # 0 = 不限步数
    lr_base=0.0018,
    lr_floor=0.0005,
    warmup=200,
    min_keep=0.65,      # 词表覆盖率门槛，低于它整窗丢弃
    checkpoint=100,     # 每 N 步落盘一次（权重+断点成对写）
    mem_soft=5500.0,
    mem_hard=7000.0,
    log_every=20,       # 每 N 步把实况行写成永久日志
    save_progress_every=5,
    nice=True,          # 降优先级，用户还要上课
    plateau=0.004,      # 一遍下来 loss 相对降幅不足 0.4% → 学到位
    quiet=True,         # 静音引擎自带启动动画
)


def _pass_seed(p: int) -> int:
    """每一遍用不同（但可复现）的洗牌种子。"""
    return (20260913 + p * 7919 + 977) & 0x7FFFFFFF


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


def _bar(done: int, total: int, width: int = 28) -> str:
    """纯 ASCII 进度条 —— 不依赖任何终端字形，Windows 控制台稳稳的。"""
    pct = 100.0 * float(done) / float(max(1, total))
    fill = int(width * pct / 100.0)
    if fill < 0:
        fill = 0
    if fill > width:
        fill = width
    return "[" + "#" * fill + "." * (width - fill) + "]"


def live_line(done: int, total: int, step: int, loss: float, el: float, eta: float,
              rss: float, tag: str = "") -> str:
    """前台实况行：进度条 + 百分比 + 窗口数 + 步数 + loss + 已用 + 剩 + 内存。"""
    return ("  {} {:5.1f}%  窗口 {:>7,}/{:<7,}  步 {:>7,}  loss {:6.3f}  "
            "已用 {}  剩 {}  RSS {:>5.0f}MB  {}").format(
        _bar(done, total), 100.0 * done / max(1, total), done, total, step,
        loss, hms(el), ("计算中…" if eta < 0 else hms(eta)), rss, tag).rstrip()


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
RESUME_PATH = os.path.join(STATUS_DIR, "study_B_resume.json")
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
    for d in CANDIDATE_CORPUS_DIRS:
        if os.path.isdir(d):
            return d
    # 兜底：在桌面 / 项目目录里找名字含"网络语料"的目录
    for root in (DESKTOP, HERE):
        try:
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p) and CORPUS_SUBDIR in name:
                    return p
                if os.path.isdir(p) and CORPUS_PARENT in name:
                    q = os.path.join(p, CORPUS_SUBDIR)
                    if os.path.isdir(q):
                        return q
        except Exception:
            pass
    raise SystemExit(
        "[×] 找不到网络语料目录（找过：{}）\n"
        "    先在桌面建《小方小说》文件夹，把爬回来的语料_*.txt 放进里面的《网络语料》子目录，"
        "或者用 --corpus 指定目录。".format(" | ".join(CANDIDATE_CORPUS_DIRS)))


def _skip_name(n: str) -> bool:
    """`.` / `_` / `~$` 开头的一律不碰 —— 状态文件、断点、临时文件绝不能当语料读。
    （爬取脚本的 .crawl_status 正好被这条挡在外面。）"""
    return n.startswith(".") or n.startswith("_") or n.startswith("~$")


def list_txt(corpus_dir: str):
    """只收 .txt，**只往下钻一层**。返回 [(显示名, 绝对路径, 字节数)]，顺序稳定。"""
    entries = []
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

    files = []
    for name, p in entries:
        try:
            sz = os.path.getsize(p)
        except Exception:
            continue
        if sz > 0:
            files.append((name, p, sz))
    if not files:
        raise SystemExit("[×] 目录里没有可用的 .txt：{}\n"
                         "    是不是还没爬？或者爬出来的语料在别的地方（用 --corpus 指过去）。"
                         .format(corpus_dir))
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
    """先扫一遍统计 token / 窗口总量与词表覆盖率，好算进度与 ETA。不驻留语料，只数数。"""
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
        if len(buf) >= 12:                      # 尾部的零头也喂一口，别浪费
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
        description="AI 小方 FlphaLit Covi 1 · 脚本 B：把网络语料喂成权重（前台可见）")
    ap.add_argument("--corpus", default="", help="语料目录（默认自动找桌面《小方小说\\网络语料》）")
    ap.add_argument("--tier", default=DEFAULTS["tier"], choices=("lite", "pro", "ultra"))
    ap.add_argument("--passes", type=int, default=DEFAULTS["passes"], help="最多学几遍")
    ap.add_argument("--hours", type=float, default=DEFAULTS["hours"], help="撑够这么多小时就收工（0=不限）")
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"], help="只跑这么多步（0=不限）")
    ap.add_argument("--window", type=int, default=DEFAULTS["window"])
    ap.add_argument("--stride", type=int, default=DEFAULTS["stride"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr_base"])
    ap.add_argument("--checkpoint", type=int, default=DEFAULTS["checkpoint"], help="每 N 步落盘一次")
    ap.add_argument("--log-every", type=int, default=DEFAULTS["log_every"], help="每 N 步永久记一行日志")
    ap.add_argument("--min-keep", type=float, default=DEFAULTS["min_keep"],
                    help="窗口的词表命中率门槛，低于它整窗丢弃（默认 0.65）")
    ap.add_argument("--mem-hard", type=float, default=DEFAULTS["mem_hard"],
                    help="硬内存上限 MB，触到就落盘退出（默认 7000）")
    ap.add_argument("--mem-soft", type=float, default=DEFAULTS["mem_soft"],
                    help="软内存上限 MB，触到先 gc 再让路（默认 5500）")
    ap.add_argument("--scan-only", action="store_true", help="只统计语料规模，不进训练")
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
    stride = max(1, min(int(args.stride), window))
    passes = max(1, int(args.passes))

    DEFAULTS["log_every"] = max(1, int(args.log_every))
    DEFAULTS["min_keep"] = min(1.0, max(0.0, float(args.min_keep)))
    DEFAULTS["mem_hard"] = float(args.mem_hard)
    DEFAULTS["mem_soft"] = float(args.mem_soft)
    DEFAULTS["checkpoint"] = max(1, int(args.checkpoint))

    # ---- 引擎：档位闸门走正式版那套（XF_TIER），日志静音由我们自己做 ----
    os.environ["XF_TIER"] = tier
    # 训练窗口闸门必须在 import 引擎之前设好（只影响训练，聊天/生成一字不变）
    os.environ["XF_TRAIN_SEQ"] = str(int(window))
    if DEFAULTS["quiet"] and not os.environ.get("XF_VERBOSE"):
        os.environ.setdefault("XF_QUIET", "1")
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    st = Status("study_B_dry" if args.dry_run else "study_B")
    if args.dry_run:
        RESUME_PATH = os.path.join(STATUS_DIR, "_dryrun_B_resume.json")   # 自检不碰正式断点
    t_start = time.time()
    st.set(state="boot", tier=tier)

    print("=" * 78)
    print(" AI 小方 · FlphaLit Covi 1 —— 脚本 B：网络语料权重学习器（前台可见）")
    print(" 档位={} · 窗口={} · 步长={} · 最多 {} 遍".format(tier.upper(), window, stride, passes))
    print(" 本进程跑在前台：这个窗口就是学习现场，进度实时刷新，绝不偷偷后台跑。")
    print("=" * 78, flush=True)

    if args.no_nice:
        pass
    elif lower_priority():
        print("[i] 已把本进程降到「低于正常」优先级，尽量不打扰上课。", flush=True)

    # ---- 语料 ----
    corpus_dir = find_corpus_dir(args.corpus)
    files = list_txt(corpus_dir)
    total_bytes = sum(f[2] for f in files)
    print("[1/4] 语料目录：{}".format(corpus_dir))
    _show = files[:12]
    for name, _p, size in _show:
        print("      · {:<26} {:>8.2f} MB".format(name, size / 1048576.0))
    if len(files) > len(_show):
        print("      · …… 还有 {} 个文件（共 {} 个）".format(len(files) - len(_show), len(files)))
    print("      合计 {:.2f} MB（只取 .txt，状态目录/断点/临时文件一律不读）"
          .format(total_bytes / 1048576.0), flush=True)
    st.set(state="corpus", corpus_dir=corpus_dir, files_count=len(files),
           corpus_mb=round(total_bytes / 1048576.0, 3))

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
        tr._train_path = lambda: os.path.join(STATUS_DIR, "_dryrun_B_train.npz")
        print("[i] dry-run：权重只写临时文件，不动正式模型。", flush=True)

    # ---- 扫描 ----
    print("[3/4] 正在统计语料规模…", flush=True)
    t0 = time.time()
    stat = scan_corpus(files, tk, M.TERMINATOR, window, stride, vocab=set(tr.token2id.keys()))
    total_tokens = sum(s["tokens"] for s in stat)
    total_windows = max(1, sum(s["windows"] for s in stat))
    for s in stat[:12]:
        print("      · {:<26} {:>11,} 字 → {:>11,} token → {:>9,} 窗口 · 命中词表 {:.1%}".format(
            s["name"], s["chars"], s["tokens"], s["windows"], s["coverage"]))
    if len(stat) > 12:
        print("      · …… 其余 {} 个文件合计 {:>11,} token".format(
            len(stat) - 12, sum(s["tokens"] for s in stat[12:])))
    print("      合计 {:,.0f} token / {:,.0f} 窗口（一遍）；统计用时 {:.1f}s".format(
        total_tokens, total_windows, time.time() - t0), flush=True)

    if args.scan_only:
        print("[i] --scan-only：只统计，不进训练。", flush=True)
        st.set(state="scan_only", total_tokens=int(total_tokens), total_windows=int(total_windows))
        return 0

    # ---- 断点 ----
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

    for sname in ("SIGINT", "SIGTERM", "SIGBREAK"):
        try:
            signal.signal(getattr(signal, sname), _on_signal)
        except Exception:
            pass

    print("[4/4] 开始学习（真实反向传播 + AdamW 调权重）…", flush=True)
    print("      注：本进程第一步要现场编译 GPU 内核，约 15 秒（只花这一次，之后每步约 1 秒），"
          "看到停住别慌，那是在编译，不是卡死。", flush=True)
    print("      下面这条实况行会原地刷新：进度条 / 百分比 / 窗口数 / 步数 / loss / 已用 / 剩余时间 / 内存。",
          flush=True)
    print("", flush=True)
    st.set(state="training", total_windows=total_windows, total_tokens=int(total_tokens),
           passes_planned=passes, lr=args.lr)

    # ---- 断点坐标：上次离开的那一窗 ----
    start_pass = max(1, min(int(resume.get("pass", 1)) if resume else 1, passes))
    cursor = (start_pass,
              int(resume.get("file_idx", 0)) if resume else 0,
              int(resume.get("win_idx", 0)) if resume else 0)

    step_all = int(resume.get("step_all", 0)) if resume else 0
    run_steps = 0
    seen_windows = int(resume.get("seen_windows", 0)) if resume else 0
    kept = int(resume.get("kept", 0)) if resume else 0
    skipped = int(resume.get("skipped", 0)) if resume else 0
    keep_sum = 0.0
    keep_cnt = 0
    best_pass = None
    pass_log = []
    recent = []            # 最近若干步 loss
    last_info = None
    last_live = 0.0
    all_done = False       # 是不是把所有遍数都学完了（学完才清断点）
    last_loss = float("nan")

    def _eta_now():
        """实时剩余时间：优先按"步数配额 / 到点时刻"，否则按"剩余窗口 × 平均每步耗时"。"""
        el = time.time() - t_start
        if hard_steps:
            return max(0.0, (hard_steps - run_steps) * (el / max(1, run_steps)))
        if deadline:
            return max(0.0, deadline - time.time())
        return max(0.0, (total_windows - seen_windows) * (el / max(1, run_steps)))

    try:
        for p in range(start_pass, passes + 1):
            if stop["flag"] or (hard_steps and run_steps >= hard_steps):
                break
            order = list(files)
            if p > 1:
                random.Random(_pass_seed(p)).shuffle(order)   # 换个顺序再学一遍
            p_loss, p_n, p_t0 = 0.0, 0, time.time()
            print("\n—— 第 {}/{} 遍 · {} 个文件 ——".format(p, passes, len(order)), flush=True)

            for fi, wi, name, toks in iter_windows(order, tk, M.TERMINATOR, window, stride):
                if stop["flag"]:
                    break
                if hard_steps and run_steps >= hard_steps:
                    break
                if (p, fi, wi) < cursor:      # 上次已经学过的窗口，跳过
                    continue
                if deadline and time.time() > deadline:
                    print("\n[i] 到点了（--hours），收工落盘。", flush=True)
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
                last_loss = loss
                p_loss += loss
                p_n += 1
                recent.append(loss)
                if len(recent) > 200:
                    del recent[:len(recent) - 200]

                # ---- 前台实况行（原地刷新，每步都动）----
                avg = sum(recent) / len(recent) if recent else 0.0
                el = time.time() - t_start
                eta = _eta_now()
                line = live_line(seen_windows, total_windows, step_all, loss, el, eta, rss_mb())
                if not sys.stdout.isatty():
                    if kept % DEFAULTS["log_every"] == 0 or run_steps <= 3:
                        print(line, flush=True)
                else:
                    sys.stdout.write("\r" + line + "\033[K")
                    sys.stdout.flush()

                # ---- 永久日志：每隔 log_every 步换行打一遍，方便回看 ----
                if kept % DEFAULTS["log_every"] == 0:
                    if sys.stdout.isatty():
                        sys.stdout.write("\n")
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
                    if wrote:
                        save_resume(sig=sig, file_idx=fi, win_idx=wi + 1, step_all=step_all,
                                    kept=kept, skipped=skipped, seen_windows=seen_windows,
                                    engine_steps=int(bank.steps), **{"pass": int(p)})
                    if sys.stdout.isatty():
                        sys.stdout.write("\n")
                    print("  [存] 步 {} · 检查点已写入 {}{}".format(
                        step_all, tr._train_path(), "" if wrote else "（失败，下次接着来）"), flush=True)

                # ---- 进度 / 心跳 ----
                if kept % DEFAULTS["save_progress_every"] == 0:
                    st.set(state="training", pass_now=p, step=step_all, kept=kept,
                           skipped=skipped, seen_windows=seen_windows,
                           loss=round(loss, 4),
                           avg_loss=round(avg, 4),
                           gnorm=round(float(info.get("gnorm", 0.0)), 4),
                           lr=round(bank.opt.lr, 6), rss_mb=round(rss_mb(), 1),
                           coverage=round(keep_sum / keep_cnt, 4) if keep_cnt else None,
                           elapsed=round(el, 1), eta_sec=round(eta, 1),
                           engine_steps=int(bank.steps))

                # ---- 内存闸门 ----
                if kept % 10 == 0:
                    m = rss_mb()
                    if DEFAULTS["mem_hard"] > 0 and m > DEFAULTS["mem_hard"]:
                        if sys.stdout.isatty():
                            sys.stdout.write("\n")
                        print("[!] RSS {:.0f}MB 触到硬上限 {:.0f}MB → 落盘后立刻退出，绝不溢出。"
                              .format(m, DEFAULTS["mem_hard"]), flush=True)
                        stop["flag"] = True
                        break
                    if DEFAULTS["mem_soft"] > 0 and m > DEFAULTS["mem_soft"]:
                        gc.collect()
                        if rss_mb() > DEFAULTS["mem_soft"]:
                            time.sleep(0.5)      # 让路，别跟上课抢

            # ---- 每遍收尾 ----
            if sys.stdout.isatty():
                sys.stdout.write("\n")
            wrote_p = False
            try:
                wrote_p = bool(tr.dump_train_state())
            except Exception:
                traceback.print_exc()
            if wrote_p:
                save_resume(sig=sig, file_idx=0, win_idx=0, step_all=step_all, kept=kept,
                            skipped=skipped, seen_windows=seen_windows,
                            engine_steps=int(bank.steps), **{"pass": int(p) + 1})
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
                    print("  [√] loss 相对只降了 {:.2%}（阈值 {:.2%}）→ 已经学到位，收工。"
                          .format(rel, DEFAULTS["plateau"]), flush=True)
                    clear_resume()
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
    print("\n" + "=" * 78)
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
    print("=" * 78, flush=True)

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
