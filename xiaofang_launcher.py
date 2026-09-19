# -*- coding: utf-8 -*-
# ============================================================
#  AI 小方 · Covi 1 启动台   (xiaofang_launcher.py)
# ------------------------------------------------------------
#  正常入口：双击同目录下的「启动小方.bat」。
#  这个 .py 故意用 ASCII 文件名 —— bat 里要写它的名字，而 cmd.exe 按"当前
#  活动代码页"解码批处理，写中文文件名很容易变成乱码导致判断失败。
#  所有中文界面都在这个文件里出（Python 源码固定按 UTF-8 解码），稳。
#
#  用法：
#     python xiaofang_launcher.py            → 弹菜单选 1/2/3/4
#     python xiaofang_launcher.py 2          → 直接开 Pro（跳过菜单）
#     python xiaofang_launcher.py --tier ultra
#     python xiaofang_launcher.py 4          → 直接进历史版本（三级页面）
#     python xiaofang_launcher.py --check    → 只体检不启动（排查用）
#     python xiaofang_launcher.py --bootstrap→ 首启自检：报型号 + 补齐依赖
#
#  v2.6 新增「4 = 历史版本」：
#    历史版本目录里一个版本经常带好几个分支（v0.4 / v0.4_beta / v0.4search …），
#    所以是三层页面 ——
#      二级：选"版本系列"（v0.4 系列 / v1.1 系列 …），键位先数字 1-9，
#            不够用接着排 QWER…（历史版本太多，数字确实不够）。
#      三级：选这一系列里的"真正版本号 / 分支"，键位同样是数字（不够补字母）。
#    「0」或「b」在二/三级里都是"退回上一层"。
# ============================================================

import os
import re
import sys
import glob
import hashlib
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "xiaofang_covi1.py")
WEB = os.path.join(HERE, "xiaofang_web.py")     # Web 启动方式的转接口（外挂的那一个 py）
CACHE_DIR = os.path.join(HERE, "权重缓存")
HIST_DIR = os.path.join(HERE, "历史版本")
# 环境体检通过之后留个记号（第一行是解释器路径），
# 下次双击「启动小方.bat」看到它还在，就直接跳过整套自举。
MARKER = os.path.join(HERE, ".xf_env_ok")

# 引擎真正会 import 的模块 → pip 上的包名
DEP_MODULES = [
    ("numpy", "numpy"),
    ("colorama", "colorama"),
    ("pyfiglet", "pyfiglet"),
    ("ddgs", "ddgs"),
    ("psutil", "psutil"),
    ("requests", "requests"),
    ("bs4", "beautifulsoup4"),
    ("flask", "flask"),          # 只有 Web 启动方式要用，但一起体检掉，免得切 Web 时才报缺
]

# 依赖清单的指纹。记号文件里也存一份 —— 以后只要往 DEP_MODULES 里加一样东西，
# 指纹就跟着变，下次启动会自动重新体检补齐。
# （flask 当初就是这么漏掉的：记号是清单还没它的时候落的，.bat 一看记号在就
#   跳过整套自举，于是新加的依赖永远等不到安装。指纹就是补这个洞的。）
DEP_SIG = hashlib.md5(
    ("|".join("%s=%s" % (m, p) for m, p in DEP_MODULES)).encode("utf-8")
).hexdigest()[:8]

# —— 三档：dim / 头数 / 层数要跟主程序里的 _MODEL_TIERS 对齐 ——
#   dim 同时也是权重缓存文件名里的那段 d<dim>，用它判断"这档有没有现成权重"。
# v3.8: 三档拉开 —— Lite≈2.07B(更快更省) / Pro≈2.40B / Ultra≈2.50B, 参数真正阶梯化。
#   注意 dim 同时也是权重缓存文件名里的那段 d<dim>, 用它判断"这档有没有现成权重"。
TIERS = [
    dict(key="lite",  no="1", name="Lite",  dim=3328, params="2.07B", heads=26, layers=15,
         tag="最快 · 日常聊天、随口问问，秒回"),
    dict(key="pro",   no="2", name="Pro",   dim=3584, params="2.40B", heads=28, layers=15,
         tag="更聪明 · 想得更深、意图识别更准"),
    dict(key="ultra", no="3", name="Ultra", dim=3660, params="2.50B", heads=30, layers=15,
         tag="最深度 · 复杂长题、要多想几轮最稳"),
]
BY_KEY = {t["key"]: t for t in TIERS}
BY_NO = {t["no"]: t for t in TIERS}

RESET = "\033[0m"
C_TITLE = "\033[96m"     # 亮青
C_HOT = "\033[93m"       # 亮黄
C_DIM = "\033[90m"       # 暗灰
C_OK = "\033[92m"        # 绿
C_WARN = "\033[93m"


def _fix_console():
    """把标准输出掰成 UTF-8，免得在老控制台上打中文直接抛 UnicodeEncodeError。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _w(s):
    """按显示宽度算长度：中文/全角算 2 格，用来对齐菜单。"""
    n = 0
    for ch in s:
        n += 2 if (ord(ch) > 0x2E80 and not (0xFE00 <= ord(ch) <= 0xFE0F)) else 1
    return n


def _pad(s, width):
    return s + " " * max(0, width - _w(s))


def cached_dims():
    """扫一眼权重缓存，返回"已经生成好权重"的 d_model 集合。

    没缓存的那一档不是不能开，只是首次要现场冷编译（大概一两分钟）。"""
    got = set()
    if not os.path.isdir(CACHE_DIR):
        return got
    for f in glob.glob(os.path.join(CACHE_DIR, "fw_v1_d*.bin")):
        base = os.path.basename(f)
        try:
            seg = base.split("_d", 1)[1].split("_", 1)[0]
            got.add(int(seg))
        except Exception:
            continue
    return got


def banner():
    line = "═" * 60
    print()
    print("  " + C_TITLE + line + RESET)
    print("   " + C_HOT + "AI 小方" + C_TITLE + "  ·  Covi 1  " + C_DIM + "启动台" + RESET)
    print("  " + C_TITLE + line + RESET)
    print()


def mode_menu():
    """首层：先问"用哪种方式启动小方"。

    默认 CLI —— 什么都不按就回车，落回 1（CLI 更成熟，贯穿诸多代）。
    """
    banner()
    print("   请选择启动方式，输入数字后回车：" + RESET)
    print()
    print("   " + C_HOT + "1" + RESET + "   " + _pad("命令行 CLI", 12)
          + C_DIM + "默认 · 原汁原味的终端窗口，键位最全，老版本也都走这条路" + RESET)
    print("   " + C_HOT + "2" + RESET + "   " + _pad("Web 界面", 12)
          + C_DIM + "本地开个端口，像应用一样单独弹一个窗口（不是浏览器标签页）" + RESET)
    print("        " + C_DIM + "└ 现代拟物 · 模型三级下拉 · 思考强度滑块 · 历史对话" + RESET)
    print()
    print("   " + C_DIM + "直接回车 = 1 (CLI)   ·   输入 q 退出" + RESET)
    print()
    while True:
        try:
            raw = input("   " + C_HOT + "请选择 ❯ " + RESET)
            raw = raw.replace("\ufeff", "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in ("q", "quit", "exit", "退出"):
            return None
        if raw == "":
            return "cli"
        if raw in ("1", "cli", "命令行", "终端", "cmd"):
            return "cli"
        if raw in ("2", "web", "网页", "界面"):
            return "web"
        print("   " + C_WARN + "请输入 1 或 2（直接回车 = 1，q 退出）。" + RESET)


def menu(cached, head=True):
    if head:
        banner()
    print("   请选择要启动的版本，输入数字后回车：" + RESET)
    print()
    for t in TIERS:
        ready = t["dim"] in cached
        mark = (C_OK + "权重已就绪 · 秒开" + RESET) if ready else (C_WARN + "首次冷编译 · 约 1~2 分钟" + RESET)
        head = "   {}  {}   {}   {}头/{}层".format(
            C_HOT + t["no"] + RESET,
            _pad(t["name"], 6),
            _pad(t["params"], 6),
            t["heads"], t["layers"])
        print(head + "   " + t["tag"])
        print("        " + C_DIM + "└ " + RESET + mark)
    nhist = len(scan_history())
    print("   " + C_HOT + "4" + RESET + "   " + _pad("历史", 6) + "   回看老版本"
          + ("（" + str(nhist) + " 个系列）" if nhist else "") + "   "
          + C_WARN + "含分支 · 进三层页面挑" + RESET)
    print("        " + C_DIM + "└ 先选版本系列，再选真正的版本号 / 分支" + RESET)
    print()
    print("   " + C_DIM + "直接回车 = 1 (Lite)   ·   4 = 历史版本   ·   输入 q 退出" + RESET)
    print()
    while True:
        try:
            raw = input("   " + C_HOT + "请选择 ❯ " + RESET)
            raw = raw.replace("\ufeff", "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in ("q", "quit", "exit", "退出"):
            return None
        if raw == "":
            return BY_NO["1"]
        if raw in ("4", "hist", "history", "历史", "历史版本"):
            return HIST_SENTINEL
        if raw in BY_NO:
            return BY_NO[raw]
        for t in TIERS:
            if raw == t["key"] or raw == t["name"].lower():
                return t
        print("   " + C_WARN + "请输入 1 / 2 / 3 / 4（或直接回车选 1，q 退出）。" + RESET)


# ============================================================
#  「4 = 历史版本」三层页面
# ------------------------------------------------------------
#  二级（系列）：把 历史版本/ 下的目录按基础版本号归并 ——
#     v0.4 / v0.4_beta / v0.4search 都归到「v0.4 系列」。
#  三级（分支）：系列里真正的版本目录，选中就启动它自己的主引擎。
#  两个页面里键位都是 1-9，不够接着排 q w e r …（历史版本实在太多）；
#     「0」或「b」= 退回上一层，「q」= 退出启动台。
# ============================================================
KEYPOOL = "123456789" + "qwertyuiop" + "asdfghjkl" + "zxcvbnm"
HIST_SENTINEL = "__hist__"
HIST_BACK = "__back__"


def _ver_key(name):
    """从目录名里抽出 (大版本, 小版本)，用来排序和归并系列。"""
    m = re.match(r"^v\s*(\d+)\s*[._]?\s*(\d*)", (name or "").lower())
    if not m:
        return (999, 999)
    return (int(m.group(1)), int(m.group(2) or 0))


def _pick_engine(d):
    """在一个历史版本目录里认主引擎，返回文件名（认不出返回 None）。

    认法：挑 xiaofang 开头、排除 _data / data_ / 下划线开头的（那些是数据
    模块和自测脚本）；再优先选文件名里含目录名的那个，比如 v1.1 目录优先
    取 xiaofang_v11.py，而不是同目录里别的辅助文件。"""
    names = [os.path.basename(p) for p in glob.glob(os.path.join(d, "xiaofang*.py"))]
    cand = [n for n in names
            if not n.startswith("_") and "_data" not in n and "data_" not in n]
    if not cand:
        return None
    want = re.sub(r"[^a-z0-9]", "", os.path.basename(d).lower())

    def rank(n):
        flat = re.sub(r"[^a-z0-9]", "", n.lower())
        return (0 if (want and want in flat) else 1, len(flat), n)
    cand.sort(key=rank)
    return cand[0]


def scan_history():
    """扫一遍 历史版本/，返回 [{"title": "v0.4 系列", "items": [...]}, ...]。"""
    if not os.path.isdir(HIST_DIR):
        return []
    groups = {}
    tools = []
    try:
        entries = sorted(os.listdir(HIST_DIR))
    except Exception:
        return []
    for name in entries:
        full = os.path.join(HIST_DIR, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("_"):
            # 下划线开头的是存档工具目录（比如 _训练脚本存档），
            # 没有主引擎，就把里面的 .bat 当成工具列到最后一组。
            try:
                bats = sorted(os.path.basename(p)
                              for p in glob.glob(os.path.join(full, "*.bat")))
            except Exception:
                bats = []
            for b in bats:
                tools.append(dict(label=b, dir=full, engine=b, kind="bat"))
            continue
        eng = _pick_engine(full)
        if not eng:
            continue
        major, minor = _ver_key(name)
        series = name if major >= 999 else "v{}.{}".format(major, minor)
        groups.setdefault(series, []).append(dict(
            label=name, dir=full, engine=eng, kind="py", series=series,
            bat=os.path.isfile(os.path.join(full, "启动小方.bat")),
            mem=len(glob.glob(os.path.join(full, "xiaofang_memory*.json"))) > 0,
            npz=len(glob.glob(os.path.join(full, "*.npz"))) > 0))
    out = []
    for series in sorted(groups, key=_ver_key):
        items = sorted(groups[series], key=lambda it: _ver_key(it["label"]))
        out.append(dict(title=series + " 系列", items=items))
    if tools:
        out.append(dict(title="训练 / 语料脚本（存档）", items=tools))
    return out


def _hist_keys(items):
    """给一页里的条目配键位，返回 {键: 条目}。"""
    km = {}
    for i, it in enumerate(items):
        if i < len(KEYPOOL):
            km[KEYPOOL[i]] = it
    return km


def _hist_series_page(groups):
    """二级页面：选版本系列。返回 系列 dict / HIST_BACK / None(=退出)。"""
    banner()
    print("   " + C_HOT + "历史版本" + RESET + C_DIM
          + "  ·  第 2 层：先挑版本系列（主菜单 = 第 1 层）" + RESET)
    print()
    km = _hist_keys(groups)
    for k, g in km.items():
        names = "、".join(it["label"] for it in g["items"])
        print("   " + C_HOT + k + RESET + "   " + _pad(g["title"], 16)
              + C_DIM + str(len(g["items"])) + " 个  " + names + RESET)
    print()
    print("   " + C_DIM + "老版本分支多，所以先挑系列，下一步才是真正的版本号。"
          + "   ·   0 / b = 回主菜单   ·   q = 退出" + RESET)
    print()
    while True:
        try:
            raw = input("   " + C_HOT + "选系列 ❯ " + RESET)
            raw = raw.replace("\ufeff", "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return HIST_BACK
        if raw in ("q", "quit", "exit", "退出"):
            return None
        if raw in ("0", "b", "back", "返回"):
            return HIST_BACK
        if raw in km:
            return km[raw]
        print("   " + C_WARN + "没有这个键位，照着上面列的键选（0 回主菜单）。" + RESET)


def _hist_branch_page(g):
    """三级页面：选这一系列里真正的版本 / branch。返回 条目 / HIST_BACK / None。"""
    banner()
    print("   " + C_HOT + g["title"] + RESET + C_DIM
          + "  ·  第 3 层：选真正的版本号 / 分支" + RESET)
    print()
    km = _hist_keys(g["items"])
    for k, it in km.items():
        if it["kind"] == "bat":
            marks = ["批处理脚本"]
        else:
            marks = [it["engine"]]
            if it["bat"]:
                marks.append("自带 启动小方.bat")
            if it["mem"]:
                marks.append("有记忆")
            if it["npz"]:
                marks.append("有训练进度")
        print("   " + C_HOT + k + RESET + "   " + _pad(it["label"], 14)
              + C_DIM + "  ·  ".join(marks) + RESET)
    print()
    print("   " + C_DIM + "0 / b = 回上一层（系列列表）   ·   q = 退出" + RESET)
    print()
    while True:
        try:
            raw = input("   " + C_HOT + "选版本 ❯ " + RESET)
            raw = raw.replace("\ufeff", "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return HIST_BACK
        if raw in ("q", "quit", "exit", "退出"):
            return None
        if raw in ("0", "b", "back", "返回"):
            return HIST_BACK
        if raw in km:
            return km[raw]
        print("   " + C_WARN + "没有这个键位，照着上面列的键选（0 返回）。" + RESET)


def hist_start(entry, pause=True):
    """启动一个历史版本（.py 走解释器，.bat 走 cmd）。"""
    path = os.path.join(entry["dir"], entry["engine"])
    if not os.path.isfile(path):
        print("   " + C_WARN + "这个版本的主文件不在了：" + path + RESET)
        return 2
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.pop("XF_TIER", None)   # 老版本没有分档概念，别把当前档位带进去
    if entry["kind"] == "bat":
        print()
        print("   " + C_OK + "▶ 正在跑 " + entry["label"] + " …" + RESET)
        print()
        try:
            rc = subprocess.call(["cmd", "/c", path], cwd=entry["dir"], env=env)
        except KeyboardInterrupt:
            print()
            rc = 130
    else:
        print()
        print("   " + C_OK + "▶ 正在启动历史版本 " + entry["label"]
              + "（" + entry["engine"] + "）…" + RESET)
        print("   " + C_DIM + "  看到那个版本的启动画面就是进去了。返回上一层继续挑别的版本。"
              + RESET)
        print()
        try:
            rc = subprocess.call([sys.executable, path], cwd=entry["dir"], env=env)
        except KeyboardInterrupt:
            print()
            print("   " + C_DIM + "已中断。" + RESET)
            rc = 130
    if pause:
        print()
        try:
            input("   " + C_DIM + "这个版本已退出。按回车回到版本列表…" + RESET)
        except (EOFError, KeyboardInterrupt):
            pass
    return rc


def hist_flow(pause=True):
    """历史版本主流程：系列 → 分支 → 启动，可一层层退回。

    返回 HIST_BACK 表示退回主菜单，None 表示用户要退出启动台。"""
    groups = scan_history()
    if not groups:
        print()
        print("   " + C_WARN + "没找到历史版本目录：" + HIST_DIR + RESET)
        print("   " + C_DIM + "按回车回主菜单。" + RESET)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        return HIST_BACK
    while True:
        g = _hist_series_page(groups)
        if g is None:
            return None
        if g == HIST_BACK:
            return HIST_BACK
        while True:
            entry = _hist_branch_page(g)
            if entry is None:
                return None
            if entry == HIST_BACK:
                break
            hist_start(entry, pause=pause)


def missing_deps():
    """返回还没装上的依赖（pip 包名）。"""
    left = []
    for mod, pkg in DEP_MODULES:
        try:
            __import__(mod)
        except Exception:
            left.append(pkg)
    return left


def ensure_deps():
    """缺依赖就自动装，别让用户自己 pip。返回 True 表示最后是齐的。"""
    missing = missing_deps()
    if not missing:
        print("   " + C_OK + "依赖已齐全，不需要安装任何东西。" + RESET)
        return True
    print("   " + C_DIM + "检测到缺少依赖：" + " ".join(missing) + "，正在自动安装…" + RESET)
    cmds = [[sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing]
    for c in cmds:
        try:
            subprocess.call(c)
        except Exception:
            pass
    left = missing_deps()
    if left:
        print("   " + C_WARN + "这些还是装不上：" + " ".join(left) + RESET)
        print("   " + C_DIM + "手动执行：pip install " + " ".join(left) + RESET)
        return False
    print("   " + C_OK + "依赖已补齐。" + RESET)
    return True


def machine_info():
    """本机型号：CPU 架构 + Windows 版本。

    启动脚本（.bat）已经探过一遍，会通过 XF_ARCH / XF_OSVER 传进来；
    万一没有（比如直接跑这个 .py），这里用 platform 兜一下。"""
    arch = (os.environ.get("XF_ARCH") or "").strip().lower()
    if not arch:
        m = (platform.machine() or "").strip().lower()
        arch = {"arm64": "arm64", "aarch64": "arm64",
                "amd64": "amd64", "x86_64": "amd64",
                "x86": "win32", "i386": "win32", "i686": "win32"}.get(m, m or "unknown")
    label = {"amd64": "x64 (AMD64)", "arm64": "ARM64", "win32": "x86 (32 位)"}.get(arch, arch)

    ver = (os.environ.get("XF_OSVER") or "").strip()
    if not ver:
        try:
            ver = ".".join(platform.win32_ver()[1].split(".")[:2])
        except Exception:
            ver = ""
    return arch, label, ver


def write_marker(arch):
    """写环境记号：第一行必须是解释器路径，.bat 要靠它判断"环境已满足"。

    行尾必须是 \\r\\n，这一点是实测出来的，别改成 \\n：cmd 的 set /p 只把
    CRLF 当换行符，碰上 LF-only 的文件它会把整份内容连同内嵌的换行一起读
    进变量，一展开就把批处理的语法打断（"The syntax of the command is
    incorrect."）。CRLF 下它老实停在第一行，也不会把 \\r 留在变量里。

    解释器路径里带 % 或 " 的极少数机器干脆不落记号 —— 让 .bat 每次都重新
    自检一遍就好，慢一点但绝不会出错。"""
    if "%" in sys.executable or '"' in sys.executable:
        return
    try:
        with open(MARKER, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(sys.executable + "\n")
            f.write(platform.python_version() + "\n")
            f.write(arch + "\n")
            f.write(DEP_SIG + "\n")      # 第 4 行：依赖清单指纹，见 DEP_SIG
    except Exception:
        pass


def deps_stale():
    """记号文件里那份依赖指纹，和现在这份清单对不上吗？

    对不上有两种可能：① 旧记号（还没写指纹，比如 flask 漏装那会儿留下的）；
    ② 清单后来加过东西。两种情况都该重新体检一遍，所以返回 True。"""
    try:
        with open(MARKER, "r", encoding="utf-8", errors="replace") as f:
            lines = [x.strip() for x in f.read().splitlines()]
    except Exception:
        return False                 # 没记号文件 = .bat 那边会走完整自举，这里不用管
    return len(lines) < 4 or lines[3] != DEP_SIG


def dep_guard():
    """启动前的最后一道闸：依赖缺了、或者清单变过，就地补齐 + 更新记号。

    早先这套只在「首启自举」里跑，而 .bat 一看到记号在就整段跳过 ——
    新加的依赖因此永远等不到安装（flask 就是这么漏的）。现在每次启动
    都过一遍这道闸，代价只是 import 几个模块。"""
    if not missing_deps() and not deps_stale():
        return True
    print()
    print("   " + C_DIM + "顺手核对一下依赖…" + RESET)
    ok = ensure_deps()
    if ok:
        write_marker(machine_info()[0])     # 指纹跟着刷新，下次就不用再补
    return ok


def bootstrap():
    """首次启动的环境自检：报型号 → 报解释器 → 补齐依赖 → 落记号。

    启动脚本卡在这一步后面，只有这里全绿了才会放出 1/2/3 选择菜单。"""
    arch, arch_label, osver = machine_info()
    line = "─" * 56
    print()
    print("   " + C_TITLE + line + RESET)
    print("   " + C_HOT + "AI 小方" + RESET + C_DIM + "  ·  首次启动 · 环境自检" + RESET)
    print("   " + C_TITLE + line + RESET)
    print()
    print("   型号    " + C_HOT + (("Windows " + osver + " · ") if osver else "") + arch_label + RESET)
    print("   Python  " + C_HOT + platform.python_version() + RESET)
    print("   位置    " + C_DIM + sys.executable + RESET)
    if not os.path.isfile(ENGINE):
        print("   主程序  " + C_WARN + "缺失：" + ENGINE + RESET)
        print()
        print("   " + C_WARN + "主程序不存在，没法启动。先把 xiaofang_covi1.py 放回这个目录再试。" + RESET)
        return 3        # P3-5: 缺主程序就别往下发了，直接给错误码
    print()

    ok = ensure_deps()
    print()
    if not ok:
        print("   " + C_WARN + "环境还有缺口，先按上面的提示补一下再启动。" + RESET)
        print()
        return 3

    write_marker(arch)
    print("   " + C_OK + "环境已就绪，接下来进入版本选择。" + RESET)
    print("   " + C_DIM + "已经记下来了，下次启动不再重复这套检查，直接弹菜单。" + RESET)
    print()
    return 0



def launch(tier, pause=True):
    if not os.path.isfile(ENGINE):
        print("   " + C_WARN + "找不到主程序：" + ENGINE + RESET)
        return 2
    env = dict(os.environ)
    env["XF_TIER"] = tier["key"]
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print()
    print("   " + C_OK + "▶ 正在启动 " + tier["name"] + " 档（XF_TIER=" + tier["key"] + "）…" + RESET)
    print("   " + C_DIM + "  看到小方的启动画面就是进去了。想换档：关掉窗口，重新双击启动小方.bat。" + RESET)
    print()
    rc = 0
    try:
        rc = subprocess.call([sys.executable, ENGINE], cwd=HERE, env=env)
    except KeyboardInterrupt:
        print()
        print("   " + C_DIM + "已中断。" + RESET)
        rc = 130
    if pause:
        print()
        try:
            input("   " + C_DIM + "小方已退出。按回车关闭本窗口…" + RESET)
        except (EOFError, KeyboardInterrupt):
            pass
    return rc


def launch_web(pause=True):
    """走 Web：把转接口拉起来，它会自己开端口 + 单独弹一个应用窗口。

    这里不再问档位 —— 选模型这一步在网页下方那排上拉菜单里选，
    所以先从默认档（Lite）起，进页面之后随时切。
    """
    if not os.path.isfile(WEB):
        print("   " + C_WARN + "找不到 Web 转接口：" + WEB + RESET)
        return 2
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print()
    print("   " + C_OK + "▶ 正在拉起 Web 界面 …" + RESET)
    print("   " + C_DIM + "  会在本机开一个口子，然后单独弹一个窗口（不是浏览器标签页）。" + RESET)
    print("   " + C_WARN + "  稍后可能看到命令提示符窗口启动，那是本地 AI 运行窗口，请不要关。" + RESET)
    print("   " + C_DIM + "  关掉网页窗口，它就跟着退了。" + RESET)
    print()
    rc = 0
    try:
        rc = subprocess.call([sys.executable, WEB, "--from-launcher"], cwd=HERE, env=env)
    except KeyboardInterrupt:
        print()
        print("   " + C_DIM + "已中断。" + RESET)
        rc = 130
    if pause:
        print()
        try:
            input("   " + C_DIM + "小方已退出。按回车关闭本窗口…" + RESET)
        except (EOFError, KeyboardInterrupt):
            pass
    return rc


def main(argv):
    _fix_console()
    args = argv[1:]
    pause = "--no-pause" not in args
    reinstall = "--reinstall-env" in args
    args = [a for a in args if a not in ("--no-pause", "--reinstall-env")]

    # P1-7: --reinstall-env = 强制重做环境自检/补齐依赖 —— 先废掉"已自检"记号,
    #   并当场立刻跑一遍 bootstrap, 不再被"记号在就跳过"挡住。
    if reinstall:
        try:
            os.remove(MARKER)
        except Exception:
            pass
        return bootstrap()

    # 首启自检：.bat 在这一步通过之后才会放出 1/2/3 菜单
    if args and args[0] in ("--bootstrap", "bootstrap", "自检"):
        return bootstrap()

    # 每次启动都过一遍依赖闸：缺了就地补，清单变过也补。
    # —— 早先只有「首启自举」才体检，.bat 一看记号在就跳过，flask 因此漏装。
    elif not (args and args[0] in ("-h", "--help", "帮助", "--check", "check", "体检")):
        dep_guard()

    cached = cached_dims()

    if args and args[0] in ("--check", "check", "体检"):
        arch, arch_label, osver = machine_info()
        banner()
        print("   型号    " + (("Windows " + osver + " · ") if osver else "") + arch_label
              + "   (" + arch + ")")
        print("   Python  : " + sys.version.split()[0])
        print("   解释器  : " + sys.executable)
        print("   主程序  : " + ("找到" if os.path.isfile(ENGINE) else "缺失") + "  " + ENGINE)
        print("   环境记号: " + ("已落" if os.path.isfile(MARKER) else "未落") + "  " + MARKER)
        print("   权重缓存: " + ("已就绪" if cached else "空") + "  d=" + (
            ",".join(str(d) for d in sorted(cached)) if cached else "无"))
        for t in TIERS:
            print("   {:<6} XF_TIER={:<6} d{:<6} {}".format(
                t["name"], t["key"], t["dim"],
                "权重已就绪" if t["dim"] in cached else "需冷编译"))
        print()
        return 0

    # 直接指定档位：python xiaofang_launcher.py 2 / --tier ultra / 4 进历史版本
    if args:
        a = args[0]
        if a in ("-h", "--help", "帮助"):
            print("用法： python xiaofang_launcher.py [1|2|3|4|lite|pro|ultra|hist|web] [--check] [--bootstrap] [--no-pause]")
            print("      不带参数 = 先问启动方式（1=CLI 默认 / 2=Web）")
            print("      web/--web = 直接走 Web 界面（模型在网页里选）")
            print("      1=Lite  2=Pro  3=Ultra  4/--hist=历史版本（三层页面）")
            return 0
        if a in ("--web", "web", "网页", "界面"):
            if not ensure_deps():
                return 3
            return launch_web(pause=pause)
        if a in ("--hist", "--history", "hist", "history", "历史", "历史版本"):
            if not ensure_deps():
                return 3
            return 0 if hist_flow(pause=pause) is None else 0
        if a == "--tier" and len(args) > 1:
            a = args[1]
        a = a.strip().lower()
        t = BY_NO.get(a) or BY_KEY.get(a)
        if t:
            if not ensure_deps():
                return 3
            return launch(t, pause=pause)
        print("   " + C_WARN + "不认识的参数：" + a + RESET)
        print("   用法： python xiaofang_launcher.py [1|2|3|4|hist] [--check]")
        return 1

    # 主菜单循环：从历史版本里「0 / b」退回来时会重新弹主菜单，
    # 不会直接把窗口关掉，用户可以接着挑别的档位。
    while True:
        # 第一层：先定启动方式（CLI / Web），什么都不按就回车 = CLI
        mode = mode_menu()
        if mode is None:
            print("   " + C_DIM + "已取消，没有启动。" + RESET)
            return 0
        if mode == "web":
            if not ensure_deps():
                return 3
            return launch_web(pause=pause)

        # 第二层：CLI 才继续问档位 / 历史版本
        t = menu(cached, head=False)
        if t is None:
            print("   " + C_DIM + "已取消，没有启动。" + RESET)
            return 0
        if t == HIST_SENTINEL:
            if not ensure_deps():
                return 3
            if hist_flow(pause=pause) is None:
                print("   " + C_DIM + "已取消，没有启动。" + RESET)
                return 0
            continue
        print()
        print("   " + C_HOT + "已选择：" + t["name"] + "（" + t["params"] + " · "
              + str(t["heads"]) + "头 · " + str(t["layers"]) + "层）" + RESET)
        if not ensure_deps():
            return 3
        return launch(t, pause=pause)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print()
        sys.exit(130)
