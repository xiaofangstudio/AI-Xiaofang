# -*- coding: utf-8 -*-
"""
小方 · Web 外挂转接口  (xiaofang_web.py)
======================================================================
一句话: 把老引擎原封不动地"端起来", 前面开一个网页当脸。

为什么这么做
------------
引擎已经有十几代了, 每一代都是一套完整的 CLI。要让它们全部出现在网页里,
最省事、也最不容易塌的做法不是"为网页重写一遍逻辑"(那等于写两套, 迟早会分叉),
而是——**只当搬运工**:

    网页输入  ──写进 stdin──▶  引擎(照常 print)
    引擎屏幕  ──从 stdout 抄──▶  网页

引擎那边只需要保证 print 还能出来就行, 一行业务逻辑都不用改。
(covi1 额外认得一个哨兵前缀, 会把"谁说的话"标清楚; 没有也不影响, 照样能跑。)

本地 CLI 也照常一起起来 —— 只不过默认最小化。想省心就直接用小方启动器选 Web;
想自己盯命令提示符, 把它还原出来就行。

关掉网页 = 关掉小方
-------------------
网页每 1.5 秒发一次心跳。这边超过 10 秒收不到, 就认为窗口没了, 顺手把引擎收掉。
所以关窗口不会留下一个"后台偷跑"的进程。

用法
----
    python xiaofang_web.py                      # 默认 Covi1 Lite
    python xiaofang_web.py --tier pro           # 换一档
    python xiaofang_web.py --engine <某版本.py> --legacy
    python xiaofang_web.py --no-window          # 只起服务, 不自动开窗口(调试用)
"""
from __future__ import annotations

import atexit
import glob
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from urllib.parse import quote

try:
    from flask import Flask, Response, jsonify, request, send_from_directory
except Exception:                                            # pragma: no cover
    # 这台机器上真没装 flask（换了解释器、或者被人 pip uninstall 了）。
    # 别让用户自己去敲 pip —— 就地补一次再 import。
    sys.stderr.write("\n[xiaofang_web] 这台机器上没装 flask，正在自动补上…\n")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "flask"])
    except Exception:
        pass
    try:
        from flask import Flask, Response, jsonify, request, send_from_directory
    except Exception:
        sys.stderr.write("\n[xiaofang_web] 还是装不上。手动来一句：\n"
                         "    \"" + sys.executable + "\" -m pip install flask\n\n")
        raise


# ══════════════════════════════════════════════════════════════════════════
# 路径与常量
# ══════════════════════════════════════════════════════════════════════════
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DEFAULT = os.path.join(HERE, "xiaofang_covi1.py")
UI_DIR = os.path.join(HERE, "Web界面")
ASSET_DIR = os.path.join(UI_DIR, "assets")
CHAT_DIR = os.path.join(HERE, "Web对话")
CACHE_DIR = os.path.join(HERE, "权重缓存")
HIST_DIR = os.path.join(HERE, "历史版本")
SHIM_DIR = os.path.join(HERE, "Web界面", "_shim")
BG_DIR = os.path.join(HERE, "背景图")          # 网页换背景用的图, 就放在项目根下

WEB_TAG = "\x01XF\x01"
IMAGE_DIR = os.path.join(os.path.dirname(HERE), "小方规划", "小游戏", "图片")

BEAT_HOLD = 10.0          # 多久没心跳就认为窗口关了(秒)
BEAT_INTERVAL_HINT = 1.5

# Covi1 三档(与 xiaofang_launcher.py 的 TIERS 对齐; 这里只用到"哪一档 / 多大")
# v3.8: 三档拉开 —— Lite≈2.07B(更快更省) / Pro≈2.40B / Ultra≈2.50B。
TIERS = [
    dict(key="lite",  no="1", name="Lite",  dim=3328, params="2.07B",
         tag="最快 · 日常聊天、随口问问，秒回"),
    dict(key="pro",   no="2", name="Pro",   dim=3584, params="2.40B",
         tag="更聪明 · 想得更深、意图识别更准"),
    dict(key="ultra", no="3", name="Ultra", dim=3660, params="2.50B",
         tag="最深度 · 复杂长题、要多想几轮最稳"),
]


# ══════════════════════════════════════════════════════════════════════════
# msvcrt 垫片
#   老引擎的输入框是"逐键读"的(msvcrt.getwch), 读的是键盘不是 stdin。
#   网页那边没法按键盘, 所以给子进程塞一个假 msvcrt: 把 getwch 接到 stdin 上。
#   这样十几代引擎谁都不用改, 就能被网页喂进去。
#   covi1 自己认得 XF_WEB、会直接走 input(), 用不到这个; 但留着对老版本是保险。
# ══════════════════════════════════════════════════════════════════════════
_SHIM_SRC = '''# -*- coding: utf-8 -*-
"""小方 Web 托管用的 msvcrt 替身(由 xiaofang_web.py 自动生成, 不用手改)。"""
import sys
import os
import types

if os.environ.get("XF_WEB"):
    _m = types.ModuleType("msvcrt")
    _m.__file__ = "<xf-web-shim>"
    _buf = []
    _eof = {"v": False}

    def _pull():
        if _buf:
            return True
        if _eof["v"]:
            return False
        try:
            line = sys.stdin.readline()
        except Exception:
            _eof["v"] = True
            return False
        if line == "":
            _eof["v"] = True
            return False
        _buf.extend(line.rstrip("\\r\\n"))
        _buf.append("\\r")          # 一行读满 = 用户按了回车
        return True

    def getwch():
        if not _buf and not _pull():
            return "\\x03"          # EOF -> 引擎自己会当成中断干净退出
        return _buf.pop(0)

    def getch():
        return getwch()

    def kbhit():
        return bool(_buf)

    _m.getwch = getwch
    _m.getch = getch
    _m.kbhit = kbhit
    sys.modules["msvcrt"] = _m

    # ---- 父进程一死就自杀 ------------------------------------------------
    #   Web 服务是父进程。窗口关掉 / 服务被强杀 / 服务自己崩了 —— 只要父进程没了,
    #   这个推理进程立刻自己退出, 绝不留孤儿在后台白烧 CPU。
    #   (实测踩过: 上一轮测试跑残的引擎没人收, 挂了 2901 秒 CPU 和一个核, 把后一次
    #    测试从 54 秒拖成 200 秒。)
    #   放在垫片里而不是引擎里, 是因为垫片十几个版本通用, 一份改动全都受益。
    def _xf_watch_parent():
        import time as _t
        if os.name != "nt":
            return
        try:
            ppid = int(os.getppid())
        except Exception:
            return
        if not ppid or ppid == os.getpid():
            return
        try:
            import ctypes as _c
            k32 = _c.windll.kernel32
            k32.OpenProcess.restype = _c.c_void_p
            k32.OpenProcess.argtypes = [_c.c_uint32, _c.c_int, _c.c_uint32]
            k32.WaitForSingleObject.argtypes = [_c.c_void_p, _c.c_uint32]
            k32.CloseHandle.argtypes = [_c.c_void_p]
            SYNCHRONIZE = 0x00100000
            h = k32.OpenProcess(SYNCHRONIZE, False, ppid)
            if not h:
                return
        except Exception:
            return
        while True:
            try:
                # 0 = 等到了 = 父进程已经收摊
                if k32.WaitForSingleObject(h, 1000) == 0:
                    k32.CloseHandle(h)
                    os._exit(0)
            except Exception:
                return

    try:
        import threading as _th
        _th.Thread(target=_xf_watch_parent, daemon=True,
                   name="xf-parent-watch").start()
    except Exception:
        pass
'''


def ensure_shim():
    try:
        os.makedirs(SHIM_DIR, exist_ok=True)
        p = os.path.join(SHIM_DIR, "sitecustomize.py")
        old = ""
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                old = f.read()
        if old != _SHIM_SRC:
            with open(p, "w", encoding="utf-8") as f:
                f.write(_SHIM_SRC)
        return SHIM_DIR
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════════
_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"       # CSI (颜色/光标)
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"   # OSC
    r"|\x1b[()][0-9A-Za-z]"            # 字符集切换
    r"|\x1b[=>NODEc]"                  # 单字符控制
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f]"   # 其余不可见控制符(含哨兵残留)
)


def strip_ansi(s):
    return _ANSI.sub("", s).replace("\r", "")


def human_size(n):
    try:
        n = float(n)
    except Exception:
        return "?"
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return ("%.0f %s" % (n, u)) if u in ("B", "KB") else ("%.1f %s" % (n, u))
        n /= 1024.0
    return "?"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# 优先用这个口子: 固定下来浏览器书签/历史才能复用, 被占了再往后顺延。
PREF_PORT = 8799


def pick_port(preferred=PREF_PORT, tries=24):
    """先试固定端口, 一路往后找, 全被占才回到随机空闲口。

    为什么要固定: 端口每次随机, 上一轮开过的那个窗口/书签就指着老地址,
    再打开就是"找不到本地端口"。固定 + 落一份端口文件, 谁都能找回来。
    """
    for i in range(max(1, int(tries))):
        p = preferred + i
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except Exception:
            try:
                s.close()
            except Exception:
                pass
            continue
    return free_port()


def port_file_paths():
    out = [os.path.join(UI_DIR, "_port.txt")]
    la = os.environ.get("LOCALAPPDATA")
    if la:
        out.append(os.path.join(la, "XFWeb", "port.txt"))
    return out


def write_port_file(port, url):
    for p in port_file_paths():
        try:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write("%d\n%s\n" % (port, url))
        except Exception:
            pass


def wait_port(port, timeout=90.0, host="127.0.0.1"):
    """死等端口真的开始收连接。

    这是"点进去提示找不到本地端口"的正主: 以前是启动后 0.8 秒盲开窗口,
    可引擎 + Flask 冷启动常常要好几秒, 窗口开出来时那个口子还没人听,
    浏览器就报连不上。改成: 端口通了才开窗, 不通就一直等。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        s.settimeout(0.4)
        try:
            s.connect((host, int(port)))
            s.close()
            return True
        except Exception:
            try:
                s.close()
            except Exception:
                pass
            time.sleep(0.12)
    return False


def open_app_window(url):
    """把网页开成一个"独立窗口"(没有标签栏/地址栏), 而不是浏览器里多一个标签页。"""
    profile = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "XFWebProfile")
    cands = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
    ]
    for exe in cands:
        if exe and os.path.isfile(exe):
            try:
                subprocess.Popen(
                    [exe, "--app=" + url, "--window-size=1280,880",
                     "--user-data-dir=" + profile,
                     "--no-first-run", "--no-default-browser-check",
                     "--disable-features=Translate,msEdgeTranslate"],
                    close_fds=True)
                return True
            except Exception:
                continue
    try:
        webbrowser.open(url)
        return False
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
# 模型清单(给网页那个三级下拉用)
#   正式版 → 先行版 1.8~1.0 → 测试版 0.9~0.1
#   有变体的展开下一级; 没有变体的点一下就直接调它。
# ══════════════════════════════════════════════════════════════════════════
_SUFFIX_CN = {
    "alpha": "Alpha", "alpha2": "Alpha 二", "alpha3": "Alpha 三",
    "beta": "Beta", "beta2": "Beta 二",
    "flash": "Flash", "flash2": "Flash 二",
    "pro": "Pro", "search": "搜索版", "tiny": "Tiny",
}


def _suf_name(suf):
    """把目录名尾巴 (alpha / _beta / flash2 …) 收拾成好看的名字; 没尾巴就返回空串。"""
    s = (suf or "").strip().strip("_-").lower()
    if not s:
        return ""
    return _SUFFIX_CN.get(s, s.upper())


def _pick_engine(folder):
    """从一个版本目录里挑出"该跑哪个 py"。"""
    best = None
    try:
        names = sorted(os.listdir(folder))
    except Exception:
        return None
    for f in names:
        if not f.endswith(".py") or f.startswith("_"):
            continue
        low = f.lower()
        if low.startswith("xiaofang_data") or low.startswith("xiaofang_memory") \
                or low.startswith("xiaofang_train") or low.startswith("xiaofang_code"):
            continue
        if not low.startswith("xiaofang"):
            continue
        if best is None or len(f) > len(best):
            best = f
    return best


def _cache_state(cwd, tier_dim=None):
    """这个模型"能不能秒开"? 有没有本地缓存? 大概多大?"""
    if tier_dim:
        hit = glob.glob(os.path.join(CACHE_DIR, "fw_v1_d%d_*.bin" % tier_dim))
        if hit:
            try:
                sz = sum(os.path.getsize(x) for x in hit)
            except Exception:
                sz = 0
            return dict(ready=True, size=human_size(sz),
                        note="已在本地缓存 · 打开秒进")
        return dict(ready=False, size="—",
                    note="首次要现场构建这一档(约十几秒), 之后一直是秒开")

    hits = []
    for pat in ("*.npz", "*.bin", "*.safetensors", "*.ckpt"):
        hits += glob.glob(os.path.join(cwd, pat))
    if not hits:
        hits += glob.glob(os.path.join(HERE, "xiaofang_train*.npz"))
    if hits:
        try:
            sz = sum(os.path.getsize(x) for x in hits)
        except Exception:
            sz = 0
        return dict(ready=True, size=human_size(sz),
                    note="带本地权重 · 直接可用(老版本启动本身偏慢)")
    return dict(ready=False, size="—",
                note="没有本地权重 · 首次启动要现算, 会明显慢一些")


def _mk_item(label, engine, cwd, extra=None):
    it = dict(id=None, label=label, engine=engine, cwd=cwd, legacy=True,
              ready=False, size="—", note="", params="", tag="")
    if extra:
        it.update(extra)
    it["id"] = "m_" + re.sub(r"\W+", "_", (os.path.relpath(engine, HERE) if engine else label)).lower()
    return it


def build_catalog():
    """整棵模型树。返回 list[一级分组]。"""
    groups = []

    # ── 正式版 ────────────────────────────────────────────────────────────
    official = []
    if os.path.isfile(ENGINE_DEFAULT):
        tiers = []
        for t in TIERS:
            st = _cache_state(HERE, t["dim"])
            tiers.append(dict(
                id="covi1_" + t["key"], label="Covi1 " + t["name"],
                desc=t["tag"], params=t["params"],
                engine=ENGINE_DEFAULT, cwd=HERE, legacy=False, tier=t["key"],
                ready=st["ready"], size=st["size"], note=st["note"],
            ))
        official.append(dict(id="covi1", label="Covi1", desc="现役正式版 · 三档可选",
                             badge="正式", children=tiers,
                             ready=tiers[0]["ready"], size=TIERS[0]["params"],
                             note="点开选 Lite / Pro / Ultra"))
    c2 = os.path.join(HIST_DIR, "Covi2合并备案_2026-09-19", "xiaofang_covi2.py")
    if os.path.isfile(c2):
        official.append(_mk_item("Covi2（合并备案）", c2,
                                 os.path.dirname(c2),
                                 dict(desc="已并入 Covi1 · 仅作存档", badge="备案")))
    groups.append(dict(id="g_official", label="正式版", desc="主力在用的", items=official))

    # ── 先行版 / 测试版: 扫历史版本目录, 按 1.x / 0.x 分家 ────────────────
    series = {}
    if os.path.isdir(HIST_DIR):
        for name in sorted(os.listdir(HIST_DIR)):
            d = os.path.join(HIST_DIR, name)
            if not os.path.isdir(d) or name.startswith("_"):
                continue
            m = re.match(r"^v(\d+)\.(\d+)([a-z0-9_]*)$", name)
            if not m:
                continue
            major, minor, suf = m.group(1), int(m.group(2)), m.group(3).lower()
            eng = _pick_engine(d)
            if not eng:
                continue
            st = _cache_state(d)
            it = _mk_item("", os.path.join(d, eng), d, dict(
                ready=st["ready"], size=st["size"], note=st["note"],
                suffix=suf, vname=_suf_name(suf),
                label=("%d.%d%s" % (int(major), minor,
                                    (" " + _suf_name(suf)) if _suf_name(suf) else "")),
            ))
            series.setdefault(major, {}).setdefault(minor, []).append(it)

    def _series_group(major, title, desc):
        buckets = series.get(major)
        if not buckets:
            return None
        items = []
        for minor in sorted(buckets.keys(), reverse=True):
            variants = buckets[minor]
            if len(variants) == 1:
                v = variants[0]
                vn = v.pop("vname", "")
                v.pop("suffix", None)
                v["label"] = "%d.%d%s" % (int(major), minor, (" " + vn) if vn else "")
                items.append(v)
            else:
                variants.sort(key=lambda x: x.get("suffix", ""))
                for v in variants:
                    vn = v.pop("vname", "")
                    v.pop("suffix", None)
                    v["label"] = "%d.%d%s" % (int(major), minor,
                                              (" " + vn) if vn else " 标准版")
                items.append(dict(
                    id="g_%s_%d" % (major, minor),
                    label="%d.%d" % (int(major), minor),
                    desc="有 %d 个变体" % len(variants),
                    children=variants,
                    ready=any(v["ready"] for v in variants),
                    size="", note="点开选具体变体",
                ))
        return dict(id="g_" + major, label=title, desc=desc, items=items)

    g1 = _series_group("1", "先行版 1.8 ~ 1.0", "正式版之前的那一拨")
    if g1:
        groups.append(g1)
    g0 = _series_group("0", "测试版 0.9 ~ 0.1", "最早的那几代")
    if g0:
        groups.append(g0)

    return groups


def flatten_catalog(groups):
    out = {}

    def walk(nodes):
        for n in nodes:
            if n.get("engine"):
                out[n["id"]] = n
            if n.get("children"):
                walk(n["children"])
            if n.get("items"):
                walk(n["items"])
    walk(groups)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 设置读写(思考强度滑块落盘到 xiaofang_settings.py)
# ══════════════════════════════════════════════════════════════════════════
SETTINGS_FILE = os.path.join(HERE, "xiaofang_settings.py")


# 网页设置面板与 CLI 的 `setting` 共用同一张表 —— 键名、默认值、取值范围全对齐，
#   任何一边改了另一边跟着变，不会再出现「网页上有的开关命令行没有」这种错位。
#   每项: (默认值, 类型, 最小, 最大)；str/bool 的 min/max 写 None
SETTING_SPEC = {
    "THINK_MODE":          ("think", "str",   None, None),
    "THINK_TURNS":         (3,       "int",   1,    8),
    "EMOTION_SENSITIVITY": (1.2,     "float", 0.0,  3.0),
    "USE_PUNCT_EMOJI":     (True,    "bool",  None, None),
    "SHOW_DEEP_THINK":     (True,    "bool",  None, None),
    "ENERGY":              (0.9,     "float", 0.0,  2.0),
    "FORCE_OFFLINE":       (False,   "bool",  None, None),
    "FLASH_LEVEL":         (2,       "int",   0,    3),
}
THINK_MODES = ("off", "think", "multi")


def _cast_lit(v):
    """把 xiaofang_settings.py 里那行 `KEY = xxx` 右边的字面量还原成 Python 值。"""
    s = (v or "").strip()
    if s[:1] in ("'", '"'):
        return s.strip("'\"")
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        f = float(s)
    except Exception:
        return s
    # 写成 3 就当 int，写成 1.2 / 1e-3 就当 float —— 轮数这类必须是整数
    if f.is_integer() and not any(c in s for c in ".eE"):
        return int(f)
    return f


def _clamp_val(key, val):
    dflt, typ, lo, hi = SETTING_SPEC[key]
    try:
        if typ == "bool":
            if isinstance(val, str):
                val = val.strip().lower() in ("1", "true", "on", "yes", "开")
            else:
                val = bool(val)
        elif typ == "int":
            val = int(round(float(val)))
        elif typ == "float":
            val = round(float(val), 2)
        else:
            val = str(val)
            if key == "THINK_MODE" and val not in THINK_MODES:
                val = dflt
    except Exception:
        return dflt
    if lo is not None:
        val = max(lo, min(hi, val))
    return val


def read_settings():
    got = {k: v[0] for k, v in SETTING_SPEC.items()}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            src = f.read()
        for k in list(got.keys()):
            m = re.search(r"^\s*%s\s*=\s*(.*)$" % re.escape(k), src, re.M)
            if not m:
                continue
            val = m.group(1).strip()
            # P1-1: 行尾注释(#)只在引号外才算 —— 值里带 # (如 'think#1') 不再被吞
            _q = None
            for i, ch in enumerate(val):
                if _q:
                    if ch == _q:
                        _q = None
                elif ch in ("'", '"'):
                    _q = ch
                elif ch == "#":
                    val = val[:i]
                    break
            got[k] = _clamp_val(k, _cast_lit(val))
    except Exception:
        pass
    return got


def write_settings(patch):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return False
    for k, v in patch.items():
        # P0-1: 用 repr 而非手工拼引号 —— 值里带 ' " # 也能写回合法 Python 文件
        lit = repr(v) if isinstance(v, str) else str(v)
        new, n = re.subn(r"^(\s*%s\s*=\s*).*?$" % k, lambda m: m.group(1) + lit, src,
                         count=1, flags=re.M)
        if n:
            src = new
        else:
            src = src.rstrip("\n") + "\n%s = %s\n" % (k, lit)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
# 会话(事件流 + 引擎进程)
# ══════════════════════════════════════════════════════════════════════════
LOCK = threading.RLock()
EVENTS = deque(maxlen=6000)
EVENT_SEQ = {"n": 0}

STATE = {
    "state": "starting",     # starting / loading / ready / working / down
    "model": "",             # 当前模型的显示名
    "model_id": "",
    "engine": ENGINE_DEFAULT,
    "cwd": HERE,
    "legacy": False,
    "tier": "lite",
    "detail": "",
}

BEAT = {"t": 0.0, "armed": False}
SHUTDOWN = {"v": False}
PROC = {"p": None}
CONV = {"cid": "", "name": "", "msgs": []}


def emit(kind, text="", **kw):
    with LOCK:
        EVENT_SEQ["n"] += 1
        ev = {"i": EVENT_SEQ["n"], "t": kind, "x": text, "ts": time.time()}
        if kw:
            ev.update(kw)
        EVENTS.append(ev)
        return ev


def set_state(s, detail=""):
    if STATE["state"] == s and STATE["detail"] == detail:
        return
    STATE["state"] = s
    STATE["detail"] = detail
    emit("state", s, detail=detail)


# ── 对话存档(只留"用户说的"和"小方说的", 思考过程一概不留) ────────────────
def _conv_path(cid):
    return os.path.join(CHAT_DIR, cid + ".json")


def conv_save():
    if not CONV["cid"]:
        return
    try:
        os.makedirs(CHAT_DIR, exist_ok=True)
        data = dict(cid=CONV["cid"], name=CONV["name"], model=STATE["model"],
                    updated=time.time(), msgs=CONV["msgs"])
        with open(_conv_path(CONV["cid"]), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def conv_new(first_text=""):
    with LOCK:
        CONV["cid"] = time.strftime("c%Y%m%d_%H%M%S") + "_%03d" % (int(time.time() * 1000) % 1000)
        CONV["name"] = (first_text.strip().replace("\n", " ")[:24] or "新对话")
        CONV["msgs"] = []
        conv_save()
        emit("conv", CONV["cid"], name=CONV["name"])


def conv_push(role, text):
    if not text:
        return
    if not CONV["cid"]:
        conv_new(text)
    with LOCK:
        if CONV["msgs"] and CONV["msgs"][-1]["role"] == role:
            CONV["msgs"][-1]["text"] += ("\n" if role == "ai" else "") + text
        else:
            CONV["msgs"].append(dict(role=role, text=text))
        if CONV["name"] in ("", "新对话") and role == "user":
            CONV["name"] = text.strip().replace("\n", " ")[:24]
    conv_save()


def conv_list():
    out = []
    try:
        for f in glob.glob(os.path.join(CHAT_DIR, "*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                out.append(dict(cid=d.get("cid", ""), name=d.get("name", "未命名"),
                                updated=d.get("updated", 0), model=d.get("model", ""),
                                turns=sum(1 for m in d.get("msgs", []) if m.get("role") == "user")))
            except Exception:
                continue
    except Exception:
        pass
    out.sort(key=lambda x: x.get("updated", 0), reverse=True)
    return out


def conv_load(cid):
    try:
        with open(_conv_path(cid), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── 引擎进程 ──────────────────────────────────────────────────────────────
class Engine:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.gen = 0

    # ---- 启动 ----------------------------------------------------------
    def start(self, engine, cwd, tier=None, legacy=False, model_label=""):
        self.stop()
        with self.lock:
            self.gen += 1
            my = self.gen

        if not os.path.isfile(engine):
            emit("log", "[xiaofang_web] 找不到引擎文件: " + engine)
            set_state("down", "引擎文件不存在")
            return False

        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["XF_WEB"] = "1"
        env["XF_QUIET"] = "1"           # 引擎不要清屏 —— 屏是网页的事
        env.pop("XIAOFANG_TRAIN", None)
        if tier and not legacy:
            env["XF_TIER"] = tier
        else:
            env.pop("XF_TIER", None)

        shim = ensure_shim()
        if shim:
            env["PYTHONPATH"] = shim + os.pathsep + env.get("PYTHONPATH", "")

        # P0-6: 直接用绝对路径, 不再依赖 cwd 恰好是引擎目录 —— 谁调都不会起错/找不到
        cmd = [sys.executable, "-u", os.path.abspath(engine)]
        si = None
        flags = 0
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 6                                  # SW_MINIMIZE
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP

        set_state("loading", "正在把模型搬进内存…")
        emit("log", "[xiaofang_web] 起引擎: %s%s" % (
            os.path.basename(engine), ("  · 档位 " + str(tier)) if tier else ""))
        try:
            p = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, close_fds=True, startupinfo=si, creationflags=flags)
        except Exception as e:
            emit("log", "[xiaofang_web] 引擎起不来: %r" % (e,))
            set_state("down", "引擎起不来")
            return False

        with self.lock:
            self.proc = p
        PROC["p"] = p
        STATE["engine"] = engine
        STATE["cwd"] = cwd
        STATE["legacy"] = bool(legacy)
        STATE["tier"] = tier or ""
        STATE["model"] = model_label or os.path.basename(engine)
        emit("model", STATE["model"], model_id=STATE["model_id"], tier=STATE["tier"],
             legacy=STATE["legacy"])

        threading.Thread(target=self._pump, args=(p, my), daemon=True,
                         name="xf-web-pump").start()
        time.sleep(0.05)
        return True

    # ---- 收 stdout -----------------------------------------------------
    def _pump(self, p, my):
        fd = p.stdout.fileno() if p.stdout else None
        buf = b""
        while True:
            if my != self.gen:
                return
            try:
                data = os.read(fd, 8192) if fd is not None else b""
            except OSError:
                data = b""
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    self._on_line(line.decode("utf-8", "replace").rstrip("\r"))
                except Exception:
                    pass
        # 进程没了
        if my == self.gen:
            try:
                p.wait(timeout=3)
            except Exception:
                pass
            set_state("down", "引擎已退出")
            emit("bye", "引擎已退出")
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass

    # ---- 单行分流 ------------------------------------------------------
    def _on_line(self, raw):
        if not raw:
            return
        if raw.startswith(WEB_TAG):
            try:
                ev = json.loads(raw[len(WEB_TAG):])
            except Exception:
                return
            k, x = ev.get("t", ""), ev.get("x", "")
            if k == "user":
                conv_push("user", str(x))
                emit("user", str(x))
                set_state("working", "")
            elif k == "say":
                conv_push("ai", str(x))
                emit("say", str(x))
            elif k == "think":
                emit("think", str(x))
            elif k == "cmd":
                emit("cmd", str(x))
            elif k == "idle":
                # v3.4 修: 引擎闲下来了 = 这一轮答完了。以前这里没有 idle 分支 ——
                #   引擎报的 idle 全都被这一句 return 无声吞掉, 于是网页的"思考中"
                #   永远灭不掉(实测就是这样)。状态交还给 ready, 再把 idle 转出去。
                set_state("ready", "")
                emit("idle", "")
            elif k == "ready":
                set_state("ready", str(x))
                emit("ready", str(x))
            elif k == "bye":
                emit("bye", "")
            return

        line = strip_ansi(raw).rstrip()
        if not line.strip():
            return
        emit("log", line)

        # 老版本没有哨兵 —— 靠引擎自己的老习惯("你: " / "小方: ")把对话捡出来
        if STATE.get("legacy"):
            if line.startswith("你: ") or line.startswith("你:"):
                t = line.split(":", 1)[1].strip()
                if t:
                    conv_push("user", t)
                    emit("user", t)
                    set_state("working", "")
            elif line.startswith("小方: ") or line.startswith("小方:"):
                t = line.split(":", 1)[1].strip()
                if t:
                    conv_push("ai", t)
                    emit("say", t)

    # ---- 送一句话进去 ---------------------------------------------------
    def feed(self, text):
        with self.lock:
            p = self.proc
        if not p or p.poll() is not None:
            emit("log", "[xiaofang_web] 引擎不在了, 这句话没送出去")
            return False
        try:
            p.stdin.write((text.replace("\r", " ").replace("\n", " ") + "\n").encode("utf-8"))
            p.stdin.flush()
            return True
        except Exception as e:
            emit("log", "[xiaofang_web] 送不进去: %r" % (e,))
            return False

    # ---- 收摊 -----------------------------------------------------------
    def stop(self):
        with self.lock:
            p, self.proc = self.proc, None
        if not p:
            return
        try:
            if p.poll() is None:
                try:
                    p.stdin.close()
                except Exception:
                    pass
                time.sleep(0.12)
            if p.poll() is None:
                p.terminate()
                time.sleep(0.2)
            if p.poll() is None:
                p.kill()
            # 兜底: 上面三步是"好好说", 万一半死不退, 连它带它可能拉起的子进程一起收。
            #   引擎是 CREATE_NEW_PROCESS_GROUP 起的独立进程组, 用 /T 保证整棵都不留。
            if p.poll() is None and os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5)
                except Exception:
                    pass
        except Exception:
            pass


ENGINE = Engine()


def shutdown(reason="窗口已关闭"):
    if SHUTDOWN["v"]:
        return
    SHUTDOWN["v"] = True
    emit("log", "[xiaofang_web] 收到退出信号(%s), 正在收掉本地推理进程…" % reason)
    ENGINE.stop()
    conv_save()
    time.sleep(0.25)
    os._exit(0)


def watchdog():
    while not SHUTDOWN["v"]:
        time.sleep(0.5)
        if not BEAT["armed"]:
            continue
        if time.time() - BEAT["t"] > BEAT_HOLD:
            shutdown("心跳断了，判定窗口已关")


# ══════════════════════════════════════════════════════════════════════════
# 背景图(网页换背景用)
#   图片就摊在项目根下的「背景图」文件夹里(和 HERE 同级), 不塞进 Web界面/assets,
#   这样用户自己往文件夹里丢一张图, 刷新网页就能选到, 不用碰代码。
# ══════════════════════════════════════════════════════════════════════════
BG_EXT = (".png", ".jpg", ".jpeg", ".webp")
BG_MAX_BYTES = 12 * 1024 * 1024        # 单张上限 12MB
BG_LIST_MAX = 200                      # 列表最多给 200 条
BG_URL_PREFIX = "/backgrounds/"
_BG_BAD = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def bg_ext_ok(name):
    """是不是认的图片名 —— 只看后缀, 大小写不敏感。"""
    return os.path.splitext(name or "")[1].lower() in BG_EXT


def bg_clean(name):
    """把上传带上来的名字收拾成"能当文件名"的样子。

    浏览器给的 filename 可能是 "C:\\a\\b.png", 也可能是 "../x.png",
    一律只留最后一段; 再把 \\ / : * ? " < > | 和控制符换成下划线。
    这一步顺手就把目录穿越堵死 —— 洗完之后名字里不可能还有路径分隔符。
    """
    s = (name or "").replace("\\", "/").split("/")[-1]
    s = _BG_BAD.sub("_", s).strip().strip(".")
    if not s:
        s = "背景图"
    stem, ext = os.path.splitext(s)
    if len(stem) > 96:                 # 名字太长 Windows 这边会写不进去
        s = stem[:96] + ext
    return s


def bg_free_path(name):
    """重名就往后面挂 _1 / _2 …(一路找到空位), 返回 (完整路径, 最终文件名)。"""
    p = os.path.join(BG_DIR, name)
    if not os.path.exists(p):
        return p, name
    stem, ext = os.path.splitext(name)
    i = 1
    while True:
        cand = "%s_%d%s" % (stem, i, ext)
        p = os.path.join(BG_DIR, cand)
        if not os.path.exists(p):
            return p, cand
        i += 1


def bg_url(name):
    return BG_URL_PREFIX + quote(name)


def bg_files():
    """背景图清单: 只列图片, 按改动时间倒序(刚传的排最前), 最多 200 条。"""
    rows = []
    try:
        names = os.listdir(BG_DIR)
    except Exception:
        return rows                      # 目录还没有 / 读不动 —— 当空处理, 不报错
    for f in names:
        if not bg_ext_ok(f):
            continue
        p = os.path.join(BG_DIR, f)
        if not os.path.isfile(p):
            continue
        try:
            mt = os.path.getmtime(p)
        except Exception:
            mt = 0.0
        rows.append((mt, f))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [dict(name=f, url=bg_url(f)) for _mt, f in rows[:BG_LIST_MAX]]


# ══════════════════════════════════════════════════════════════════════════
# Flask 应用
# ══════════════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=None)
app.config["JSON_AS_ASCII"] = False


@app.after_request
def _no_cache(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def _index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/api/boot")
def _boot():
    st = read_settings()
    cat = build_catalog()
    return jsonify(dict(
        ok=True,
        state=STATE["state"], model=STATE["model"], model_id=STATE["model_id"],
        tier=STATE["tier"], engine=os.path.basename(STATE["engine"]),
        legacy=STATE["legacy"],
        catalog=cat,
        think=dict(mode=st.get("THINK_MODE", "think"), turns=st.get("THINK_TURNS", 3),
                   level=st.get("FLASH_LEVEL", 2)),
        settings=st,
        beat=int(BEAT_INTERVAL_HINT * 1000),
        hold=BEAT_HOLD,
        since=EVENT_SEQ["n"],
        conv=dict(cid=CONV["cid"], name=CONV["name"]),
    ))


@app.route("/api/stream")
def _stream():
    try:
        since = int(request.args.get("since", "0"))
    except Exception:
        since = 0

    def gen():
        idx = since
        last_ping = time.time()
        yield ": 小方 web stream\n\n"
        while not SHUTDOWN["v"]:
            with LOCK:
                backlog = [e for e in EVENTS if e["i"] > idx]
            if backlog:
                for e in backlog:
                    idx = e["i"]
                    yield "data: " + json.dumps(e, ensure_ascii=False) + "\n\n"
                last_ping = time.time()
            else:
                time.sleep(0.1)
                if time.time() - last_ping > 12:
                    last_ping = time.time()
                    yield ": ping\n\n"
        yield "event: bye\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


@app.route("/api/beat", methods=["POST"])
def _beat():
    BEAT["t"] = time.time()
    BEAT["armed"] = True
    return jsonify(ok=True)


@app.route("/api/bye", methods=["POST"])
def _bye():
    """网页在 pagehide 时会 ping 一下这里 —— 相当于"我要走了"。

    不立刻退出: 只把心跳的账做旧一点点, 让 watchdog 大约 3 秒后动手。
    这样刷新页面(几秒内会重新 /api/beat)不会误杀, 真关窗口则会很快跟着退。
    """
    if BEAT["armed"]:
        BEAT["t"] = min(BEAT["t"], time.time() - BEAT_HOLD + 3.0)
    return ("", 204)


@app.route("/api/say", methods=["POST"])
def _say():
    d = request.get_json(silent=True) or {}
    text = str(d.get("text", "")).strip()
    if not text:
        return jsonify(ok=False, err="空消息")
    if not CONV["cid"]:
        conv_new(text)
    ok = ENGINE.feed(text)
    return jsonify(ok=ok)


@app.route("/api/stop", methods=["POST"])
def _stop():
    """打断当前这一轮(等同在 CLI 里按 Ctrl+C / 发 /stop)。"""
    ENGINE.feed("/stop")
    return jsonify(ok=True)


@app.route("/api/model", methods=["POST"])
def _model():
    d = request.get_json(silent=True) or {}
    mid = str(d.get("id", ""))
    cat = flatten_catalog(build_catalog())
    item = cat.get(mid)
    if not item:
        return jsonify(ok=False, err="没有这个模型")
    STATE["model_id"] = mid
    emit("loading", "正在切换到 %s …" % item.get("label", mid))
    ENGINE.start(item["engine"], item["cwd"], tier=item.get("tier"),
                 legacy=bool(item.get("legacy")), model_label=item.get("label", mid))
    return jsonify(ok=True)


@app.route("/api/think", methods=["POST"])
def _think():
    d = request.get_json(silent=True) or {}
    mode = str(d.get("mode", "think"))
    try:
        turns = max(1, min(8, int(d.get("turns", 3))))
    except Exception:
        turns = 3
    write_settings({"THINK_MODE": mode, "THINK_TURNS": turns})
    if mode == "off":
        ENGINE.feed("/off")
    elif mode == "think":
        ENGINE.feed("/think")
    else:
        ENGINE.feed("/multi %d" % turns)
    return jsonify(ok=True, mode=mode, turns=turns)


# ── 设置面板：和 CLI 里敲 `setting` 改的是同一批开关、同一个文件 ─────────
def _setting_cmds(want):
    """把设置项翻成引擎认的那几句 /setting 子命令。"""
    out = []
    if "THINK_MODE" in want:
        out.append("/setting " + want["THINK_MODE"])
    if "THINK_TURNS" in want:
        out.append("/setting 轮数 %d" % want["THINK_TURNS"])
    if "EMOTION_SENSITIVITY" in want:
        out.append("/setting 情绪 %s" % want["EMOTION_SENSITIVITY"])
    if "ENERGY" in want:
        out.append("/setting 活泼 %s" % want["ENERGY"])
    if "FLASH_LEVEL" in want:
        out.append("/setting 速度 %d" % want["FLASH_LEVEL"])
    if "USE_PUNCT_EMOJI" in want:
        out.append("/setting 标点 " + ("on" if want["USE_PUNCT_EMOJI"] else "off"))
    if "SHOW_DEEP_THINK" in want:
        out.append("/setting 深度 " + ("on" if want["SHOW_DEEP_THINK"] else "off"))
    if "FORCE_OFFLINE" in want:
        out.append("/setting 离线 " + ("on" if want["FORCE_OFFLINE"] else "off"))
    return out


@app.route("/api/settings")
def _settings_get():
    return jsonify(ok=True, settings=read_settings())


@app.route("/api/settings", methods=["POST"])
def _settings_post():
    d = request.get_json(silent=True) or {}
    want = {}
    for k, v in d.items():
        if k in SETTING_SPEC:
            want[k] = _clamp_val(k, v)
    if not want:
        return jsonify(ok=False, err="没有认得出的设置项"), 400

    # 落盘由网页这一侧同步做完（马上读回来就是新值，界面不会回弹）；
    #   同时把等价的 /setting 喂给引擎，让它当场热更新，不必重启。
    ok = write_settings(want)
    fed = 0
    for c in _setting_cmds(want):
        if ENGINE.feed(c):
            fed += 1
    if not ok and not fed:
        return jsonify(ok=False, err="写不进去，配置文件可能被占用"), 500
    return jsonify(ok=True, hot=fed > 0, settings=read_settings())


# ── 产物夹：右边那栏"我到底给你做出来过什么" ───────────────────────────
# 引擎画完的 SVG、配套的提示词、写长了落盘的代码，统统丢在这个文件夹里。
# 这一栏只读不写：网页不生成东西，只负责把已经落盘的东西摆出来给你看。
ART_DIR = os.path.join(HERE, "产物")

# 分三档：能直接当图看的 / 能当文本读的 / 剩下的只能拿去下载
ART_IMG_EXT = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico"}
ART_TXT_EXT = {".txt", ".md", ".py", ".js", ".ts", ".css", ".html", ".htm", ".json",
               ".csv", ".log", ".xml", ".yaml", ".yml", ".bat", ".ps1", ".sh", ".ini",
               ".cpp", ".cc", ".cxx", ".h", ".hpp", ".c", ".java", ".go", ".rs", ".rb",
               ".php", ".lua", ".sql", ".kt", ".swift"}


def _art_abs(name):
    """把前端传来的名字钉死在产物夹里 —— 带 .. 或者指到外头去的一律不认。"""
    name = str(name or "").replace("\\", "/").strip().lstrip("/")
    if not name or ".." in name.split("/"):
        return ""
    root = os.path.abspath(ART_DIR)
    p = os.path.abspath(os.path.join(root, name))
    return p if (p == root or p.startswith(root + os.sep)) else ""


@app.route("/api/artifacts")
def _artifacts():
    items = []
    try:
        names = os.listdir(ART_DIR)
    except Exception:
        names = []
    for n in names:
        p = os.path.join(ART_DIR, n)
        if not os.path.isfile(p):
            continue
        try:
            stt = os.stat(p)
        except Exception:
            continue
        ext = os.path.splitext(n)[1].lower()
        kind = "img" if ext in ART_IMG_EXT else ("text" if ext in ART_TXT_EXT else "other")
        items.append(dict(name=n, size=stt.st_size, mtime=int(stt.st_mtime), kind=kind))
    # 新的排前头：刚生成的东西应该第一眼就看见
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(ok=True, dir=ART_DIR, items=items)


@app.route("/api/artifact/<path:name>")
def _artifact(name):
    p = _art_abs(name)
    if not p or not os.path.isfile(p):
        return jsonify(ok=False, err="没有这个产物"), 404
    folder, fn = os.path.split(p)
    resp = send_from_directory(folder, fn, as_attachment=bool(request.args.get("dl")))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/artifact/save", methods=["POST"])
def _artifact_save():
    """Web 端把一段代码存进产物夹。ext 若不认就退回 .txt；
    name 只拼时间戳，路径全由后端钉住，防穿越。"""
    d = request.get_json(silent=True) or {}
    text = str(d.get("text", ""))
    if not text.strip():
        return jsonify(ok=False, err="空的，没存")
    ext = str(d.get("ext", "")).strip().lstrip(".") or "txt"
    if "." in ext or "/" in ext or "\\" in ext or not ext.isalnum():
        ext = "txt"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fn = "代码_%s.%s" % (stamp, ext)
    try:
        os.makedirs(ART_DIR, exist_ok=True)
        with open(os.path.join(ART_DIR, fn), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        return jsonify(ok=False, err="写盘失败：%s" % e)
    return jsonify(ok=True, name=fn)


@app.route("/api/history")
def _history():
    return jsonify(ok=True, items=conv_list(), cur=CONV["cid"])


@app.route("/api/history/open", methods=["POST"])
def _history_open():
    d = request.get_json(silent=True) or {}
    cid = str(d.get("cid", ""))
    doc = conv_load(cid)
    if not doc:
        return jsonify(ok=False, err="读不到这条对话")
    CONV["cid"] = doc.get("cid", cid)
    CONV["name"] = doc.get("name", "未命名")
    CONV["msgs"] = doc.get("msgs", [])
    emit("conv", CONV["cid"], name=CONV["name"], msgs=CONV["msgs"])
    return jsonify(ok=True)


@app.route("/api/history/new", methods=["POST"])
def _history_new():
    conv_new("")
    emit("conv", CONV["cid"], name=CONV["name"], msgs=[])
    return jsonify(ok=True, cid=CONV["cid"])


@app.route("/api/history/del", methods=["POST"])
def _history_del():
    d = request.get_json(silent=True) or {}
    cid = str(d.get("cid", ""))
    try:
        os.remove(_conv_path(cid))
    except Exception:
        return jsonify(ok=False)
    if cid == CONV["cid"]:
        conv_new("")
    return jsonify(ok=True)


@app.route("/api/history/edit", methods=["POST"])
def _history_edit():
    """气泡上那三颗小按钮的落点。
    op=del   只拿掉这一条；
    op=revoke 撤回到本条刚发出那会儿 —— 把这条和它以后的响应一起删干净。
    索引指的就是 CONV["msgs"] 的下标（user/ai 气泡的顺序，跟 DOM 完全一致）。
    改完重新 emit('conv')，前端照最新真相整屏重排，不怕流式时串位。"""
    d = request.get_json(silent=True) or {}
    op = d.get("op")
    try:
        idx = int(d.get("idx", -1))
    except Exception:
        return jsonify(ok=False)
    msgs = CONV["msgs"]
    if not (0 <= idx < len(msgs)):
        return jsonify(ok=False)
    if op == "del":
        del msgs[idx]
    elif op == "revoke":
        del msgs[idx:]
    else:
        return jsonify(ok=False)
    if CONV["msgs"][-1:]:
        CONV["name"] = CONV["msgs"][0]["text"].strip().replace("\n", " ")[:24] or "新对话"
    else:
        CONV["name"] = "新对话"
    conv_save()
    emit("conv", CONV["cid"], name=CONV["name"], msgs=CONV["msgs"])
    return jsonify(ok=True)


@app.route("/api/quit", methods=["POST"])
def _quit():
    threading.Thread(target=shutdown, args=("用户点了退出",), daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/ping")
def _ping():
    """给前端探活用的最小接口: 能返回就说明口子是活的。"""
    return jsonify(ok=True, state=STATE.get("state"), model=STATE.get("model"))


@app.route("/api/about")
def _about():
    """这个窗口在跑的到底是什么 —— 正式版专区的信息卡用。

    全部是本地只读探测, 不碰引擎。
    """
    def _sz(p):
        try:
            n = float(os.path.getsize(p))
        except Exception:
            return 0, "—"
        u = ["B", "KB", "MB", "GB"]
        i = 0
        while n >= 1024 and i < len(u) - 1:
            n /= 1024.0
            i += 1
        return int(n) if i == 0 else n, ("%d %s" % (n, u[i]) if i == 0 else "%.2f %s" % (n, u[i]))

    out = {
        "ok": True,
        "engine": os.path.basename(STATE.get("engine") or ""),
        "engine_path": STATE.get("engine") or "",
        "model": STATE.get("model") or "",
        "model_id": STATE.get("model_id") or "",
        "tier": STATE.get("tier") or "",
        "legacy": bool(STATE.get("legacy")),
        "state": STATE.get("state"),
        "official": bool(STATE.get("model_id")),
    }

    npz = []
    try:
        for f in sorted(os.listdir(HERE)):
            if not (f.startswith("xiaofang_train_") and f.endswith(".npz")):
                continue
            p = os.path.join(HERE, f)
            it = {"file": f, "size": _sz(p)[1], "mtime": time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))}
            try:
                import numpy as _np
                z = _np.load(p, allow_pickle=True)
                if "__meta__" in z.files:
                    m = z["__meta__"]
                    it["steps"] = int(_np.asarray(m).reshape(-1)[0])
                if "__loss__" in z.files:
                    l = _np.asarray(z["__loss__"], dtype="float64").reshape(-1)
                    it["last_loss"] = round(float(l[-1]), 2) if l.size else None
                z.close()
            except Exception:
                pass
            npz.append(it)
    except Exception:
        pass
    out["train"] = npz

    try:
        bins = [f for f in os.listdir(CACHE_DIR) if f.endswith(".bin")]
        tot = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in bins)
        out["cache"] = {"files": len(bins), "size": "%.2f GB" % (tot / 1073741824.0)}
    except Exception:
        out["cache"] = {"files": 0, "size": "—"}

    try:
        out["chats"] = len([f for f in os.listdir(CHAT_DIR) if f.endswith(".json")])
    except Exception:
        out["chats"] = 0
    return jsonify(out)


@app.route("/api/backgrounds")
def _backgrounds():
    """列「背景图」里能当背景的图片(刚传的排前面)。"""
    return jsonify(ok=True, dir="背景图", files=bg_files())


@app.route("/backgrounds/<path:name>")
def _backgrounds_file(name):
    """把「背景图」里的某一张贴出去。

    防目录穿越: 先按上传那套规则洗一遍, 洗完和原来对不上(说明带了 ../ 或子目录)
    就直接 404; 末尾的 send_from_directory 自己还有一道 safe_join, 双保险。
    """
    safe = bg_clean(name)
    if safe != name or not bg_ext_ok(safe):
        return ("没有这张背景图", 404)
    return send_from_directory(BG_DIR, safe)


@app.route("/api/background/upload", methods=["POST"])
def _backgrounds_upload():
    """收背景图: multipart 里字段名叫 file, 一次可以塞好几个。

    只收 .png/.jpg/.jpeg/.webp, 单张 ≤ 12MB。文件名只取 basename 并洗掉非法字符,
    重名自动挂 _1/_2。格式或大小有一项不合格就整单不收(ok:false), 不留半截。
    """
    up = [f for f in request.files.getlist("file") if f and f.filename]
    if not up:
        return jsonify(ok=False, err="没收到文件(字段名要叫 file)")

    plan = []
    for f in up:
        raw = f.filename
        name = bg_clean(raw)
        if not bg_ext_ok(name):
            return jsonify(ok=False,
                           err="不认这个格式: %s(只收 .png/.jpg/.jpeg/.webp)" % raw)
        n = -1
        try:                             # 先问流要长度, 免得整包吞进内存
            f.stream.seek(0, os.SEEK_END)
            n = f.stream.tell()
            f.stream.seek(0)
        except Exception:
            n = -1
        if n > BG_MAX_BYTES:
            return jsonify(ok=False,
                           err="%s 太大(%s), 单张上限 12MB" % (name, human_size(n)))
        plan.append((f, name, n))

    try:
        os.makedirs(BG_DIR, exist_ok=True)
    except Exception as e:
        return jsonify(ok=False, err="建不了 背景图 文件夹: %r" % (e,))

    saved = []
    for f, name, n in plan:
        path, final = bg_free_path(name)
        try:
            if n >= 0:
                f.save(path)
            else:                        # 流不让问长度(少见): 就地数一遍再落盘
                data = f.read(BG_MAX_BYTES + 1)
                if len(data) > BG_MAX_BYTES:
                    return jsonify(ok=False,
                                   err="%s 超过单张 12MB 上限" % name)
                with open(path, "wb") as out:
                    out.write(data)
        except Exception as e:
            return jsonify(ok=False, err="写不进 背景图/%s: %r" % (name, e))
        try:
            sz = human_size(os.path.getsize(path))
        except Exception:
            sz = "?"
        saved.append(dict(name=final, url=bg_url(final), size=sz))

    return jsonify(ok=True, saved=saved, files=bg_files())


@app.route("/<path:fn>")
def _static(fn):
    # P1-3: 未知的 /api/* 应返回 JSON 404, 别落到静态文件那套 HTML 404
    if request.path.startswith("/api/"):
        return jsonify(ok=False, err="找不到这个接口"), 404
    return send_from_directory(UI_DIR, fn)


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--tier", default="lite", choices=[t["key"] for t in TIERS])
    ap.add_argument("--engine", default="")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--from-launcher", action="store_true")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    os.makedirs(CHAT_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)          # 背景图目录, 没有就现建一个空的
    if not os.path.isdir(UI_DIR):
        sys.stderr.write("[xiaofang_web] 少了界面目录: %s\n" % UI_DIR)
        return 2

    engine = a.engine or ENGINE_DEFAULT
    label = a.label or ("Covi1 " + a.tier.capitalize() if not a.legacy
                        else os.path.splitext(os.path.basename(engine))[0])
    STATE["model_id"] = ("covi1_" + a.tier) if not a.legacy else ""
    STATE["model"] = label

    port = a.port or pick_port(PREF_PORT)
    url = "http://127.0.0.1:%d/" % port
    write_port_file(port, url)

    # 引擎放到后台线程里拉: 大模型冷启动十秒上下, 以前是"先等引擎、再开 Flask",
    #   于是窗口开出来时口子还没人听 → 报"找不到本地端口"。
    #   现在 Flask 先起来, 页面先亮, 遮罩里显示进度, 引擎好了再 ready。
    threading.Thread(
        target=ENGINE.start,
        args=(engine, os.path.dirname(os.path.abspath(engine)) or HERE),
        kwargs=dict(tier=None if a.legacy else a.tier, legacy=a.legacy,
                    model_label=label),
        name="xf-web-engine", daemon=True).start()

    threading.Thread(target=watchdog, daemon=True, name="xf-web-watch").start()
    atexit.register(ENGINE.stop)

    # 这个黑窗口被"点叉"关掉时, Windows 是直接 TerminateProcess —— finally / atexit
    #   都来不及跑。真正兜底的是垫片里那条"父进程一死就自杀"(见 _SHIM_SRC)。
    #   这里只是把能被接住的几种信号也接住, 好在退出前把当前对话存一把。
    def _on_signal(signum, frame):
        shutdown("收到信号 %s" % signum)
    for _name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        try:
            signal.signal(getattr(signal, _name), _on_signal)
        except Exception:
            pass

    if not a.no_window:
        def _open_when_ready():
            if wait_port(port, timeout=120.0):
                open_app_window(url)
            else:
                sys.stderr.write("[xiaofang_web] 口子一直没起来, 请手动打开: " + url + "\n")
        threading.Thread(target=_open_when_ready, daemon=True,
                         name="xf-web-window").start()

    banner = "  小方 Web 已就绪  →  " + url
    try:
        sys.stdout.write(banner + "\n"
                         "  (这个黑窗口就是小方的本地推理进程, 请不要关掉;\n"
                         "   关掉网页窗口, 这边会自动跟着退。)\n")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        # v3.8.1: werkzeug 起服务时会往 stderr 怼一条全英文横幅 ——
        #   "WARNING: This is a development server. Do not use it in a production
        #    deployment..." + " * Running on http://127.0.0.1:x"。措辞极具恐吓感
        #   (好像哪里不安全/要出事), 容易让用户自己吓自己。上面那行中文就绪语已经
        #   把该说的都说了, 这里把 werkzeug 自己的启动日志整个静音, 清净到底。
        from werkzeug import serving as _serve
        def _quiet(level, msg, *args):   # 吞掉 werkzeug 的英文横幅/访问日志
            return
        _serve._log = _quiet
    except Exception:
        pass

    try:
        app.run(host="127.0.0.1", port=port, threaded=True,
                debug=False, use_reloader=False)
    finally:
        ENGINE.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
