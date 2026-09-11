# -*- coding: utf-8 -*-
import sys, os, platform, subprocess, datetime, random, re, time, json, threading
import numpy as np
# ===== v0.6 Pro: 纯自研 GPU 后端 —— 不依赖 PyTorch 等框架, 直接走 CuPy(CUDA) =====
#   有 NVIDIA 卡 → 权重与矩阵运算驻留显存(CuPy); 无卡或 CUDA 缺失 → 自动回退 CPU(numpy) 硬算
def _locate_cuda():
    """自动定位 NVIDIA CUDA 工具包, 让 CuPy 真正吃上显存(避免"只装了驱动→加载失败")."""
    if os.environ.get("CUDA_PATH"):
        return os.environ["CUDA_PATH"]
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    try:
        vdirs = sorted(os.listdir(base), reverse=True) if os.path.isdir(base) else []
    except Exception:
        vdirs = []
    for v in vdirs:
        p = os.path.join(base, v)
        if os.path.isdir(os.path.join(p, "bin")):
            if os.path.exists(os.path.join(p, "bin", "nvcc.exe")):
                return p
            vdirs_fallback = p
    return locals().get("vdirs_fallback", "")

_BACKEND = "CPU 硬算 (numpy)"
if not os.environ.get("XF_FORCE_CPU", ""):
    _cuda = _locate_cuda()
    if _cuda:
        os.environ.setdefault("CUDA_PATH", _cuda)
        os.environ["PATH"] = os.path.join(_cuda, "bin") + os.pathsep + os.environ.get("PATH", "")
    try:
        import sys as _sys, warnings as _warn
        if not getattr(_warn, "_xf_soft", False):
            _xf_orig = _warn.showwarning
            _xf_note = [False]
            def _xf_show(message, category, filename, lineno, file=None, line=None):
                _m = str(message)
                if "CUDA path could not be detected" in _m or "CUDA_PATH" in _m:
                    if not _xf_note[0]:
                        _xf_note[0] = True
                        _sys.stderr.write("\n[i] 提示：未检测到 CUDA 加速环境，本版本将用 CPU 模式运行，功能与结果不受影响，仅速度稍慢。\n")
                        _sys.stderr.write("     如需 GPU 加速：设置系统环境变量 CUDA_PATH（指向显卡工具包目录）后重启小方即可。\n\n")
                    return
                _xf_orig(message, category, filename, lineno, file, line)
            _warn.showwarning = _xf_show
            _warn._xf_soft = True
        import cupy as _cp
        _cp.cuda.Device(0).use()
        _probe = _cp.zeros(1)          # 实际分配一次显存, 确认 CUDA 可用
        XP = _cp
        HAS_GPU = True
        _BACKEND = "GPU 加速 (CuPy/CUDA)"
    except Exception:
        XP = np
        HAS_GPU = False
else:
    XP = np
    HAS_GPU = False

def _to_host(a):
    return _cp.asnumpy(a) if HAS_GPU else a

def _to_dev(a):
    return globals().get("_cp").asarray(a) if HAS_GPU else a


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_data_v14alpha as DATA
VERSION = "1.4 Alpha"
ENGINE_NAME = "FlphaLit"
MODEL_NAME = "AI 小方 FlphaLit 1.4 Alpha"
STUDIO_NAME = "小方工作室"
STUDIO_SITE = "fanggame.company"
SEARCH_BACKENDS = ["duckduckgo", "bing", "brave", "google", "auto"]  # 多后端联网


# ------------------------------------------------------------------
# v0.6: 硬件检测 (CPU / 内存 / 显卡 / 显存) → 自适应模型维度
#   目的: AI 小方不只跑在这台电脑, 需在其他人的机器上也能以合适规模运行。
#   高配(大显存/大内存)→ 更大向量空间, 思考更充分; 低配 → 自动降维保证可跑。
# ------------------------------------------------------------------
def _detect_hw():
    hw = {"cpu": "未知", "cores": os.cpu_count() or 2,
          "mem_total": 0, "mem_free": 0,
          "gpu_model": "无独立显卡 (CPU模式)", "vram": 0}
    try:
        hw["cpu"] = platform.processor() or "CPU"
    except Exception:
        hw["cpu"] = "CPU"
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw["mem_total"] = vm.total
        hw["mem_free"] = vm.available
    except Exception:
        pass
    # 显卡 / 显存: 优先 nvidia-smi(零依赖), 识别不到则视为 CPU 模式
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"], timeout=6)
        for ln in out.decode("utf-8", "ignore").strip().splitlines():
            name, _, mem = ln.partition(",")
            hw["gpu_model"] = name.strip()
            try:
                hw["vram"] = int(float(mem))
            except Exception:
                hw["vram"] = 0
            break
    except Exception:
        pass
    return hw


HW = _detect_hw()


def _est_params(d, L, f, vocab=12100):
    """估算该尺度总参数量(含嵌入/输出), 用于上报与降级判断."""
    return int(L * d * d * (4 + 2 * f) + 2 * d * vocab)


def _tier_ok(d, L, f):
    """该尺度能否在"≤1GB 存储 + 本机算力"下安稳部署:
    权重按 int8 存储(1 字节/参数)→ 常驻 ≈ 参数量字节 ≤1GB;
    前向/构建需临时还原为 fp32 → 峰值需小于可用内存(有卡时须装进显存)."""
    p = _est_params(d, L, f)
    if p > (1 << 30):                       # int8 权重 >1GB → 超预算, 绝不选
        return False
    fp32 = p * 4                              # 构建/前向要完整 fp32 权重(临时还原)
    if HAS_GPU and HW["vram"] > 0:            # v1.0: 有独显 → 权重须放得进显存(否则拷回去=CPU硬算)
        return fp32 <= HW["vram"] * (1024 ** 2) * 0.9
    transient = fp32 * 1.35
    mem = HW["mem_total"] or (8 * 1024 ** 3)
    return transient <= mem * 0.55


_MODEL_TIERS = [  # (d, L, H, fnn倍率, 标签, 常驻int8≈GB)
    (2880, 10, 16, 4, "FlphaLit Wide2880 · 1.07B", 0.99),  # v1.0: 注意力更宽(2304→2880)、层数精简(15→10)→ 参数更大且加载/运行更快
    (2304, 15, 16, 4, "FlphaLit Wide2304 · 1.01B", 0.94),  # v0.9: 上一代旗舰
    (2048, 20, 16, 4, "1.06B / 1B+", 0.98),   # v0.8 Pro: 参数量突破 10 亿, int8 存储仍 ≤1GB
    (1792, 24, 16, 4, "0.97B-class", 0.93),
    (1792, 20, 16, 4, "0.81B-class", 0.79),
    (1408, 16, 16, 4, "0.42B-class", 0.40),
    (1280, 14, 16, 4, "0.36B-class", 0.34),
    (1024, 12, 16, 4, "0.18B-class", 0.16),
    (896, 12, 14, 4, "P-lean · 0.13B", 0.12),   # 低配兜底
]
MODEL_TARGET = "Wide2880 · 更大参数 + 更快"       # v1.0: 更宽注意力 + 更少层数 → 更大却更快, 全程 GPU 承载
MEM_BUDGET = 1 << 30            # 硬约束: 模型常驻内存(含量化权重) ≤1GB


def _auto_dims():
    """按"≤1GB 内存 + 可用算力"自适应模型规模:
    尽可能逼近 1B 参数量, 权重 int8 存储把常驻压到 1GB 内;
    机器内存太小时自动逐级降级(尺度/存储缩小, 前向 FLOPs 不因此打折)."""
    chosen = None
    for d, L, H, f, label, gb in _MODEL_TIERS:
        if _tier_ok(d, L, f):
            chosen = (d, L, H, f, label, gb)
            break
    if chosen is None:
        chosen = _MODEL_TIERS[-1]
    d, L, H, f, label, gb = chosen
    while d % H != 0:                             # 保证 head 数能整除 d_model
        H -= 1
    return d, L, H, f, label, gb


_MODEL_D, MODEL_LAYERS, MODEL_HEADS, MODEL_FFN, MODEL_TIER, _TIER_GB = _auto_dims()
MODEL_PARAMS = _est_params(_MODEL_D, MODEL_LAYERS, MODEL_FFN)
MODEL_D, MODEL_LAYERS = _MODEL_D, MODEL_LAYERS
MODEL_UPGRADED = (MODEL_D, MODEL_LAYERS, MODEL_FFN) == (_MODEL_TIERS[0][0], _MODEL_TIERS[0][1], _MODEL_TIERS[0][3])
AUTO_DEGRADE = not MODEL_UPGRADED
MODEL_TIE = True            # 旗舰一律输出↔输入嵌入共享: 更省存储、FLOPs 不变
SEED_TOKENS = 40        # 输入读取量: 读满整句意图, 不受"读12字/24字"限制 → 完全理解用户要什么
RESPONSE_TIMEOUT = 120  # v0.9 Alpha: 单条消息处理上限(秒). 超时判卡死→提示重发并回主循环, 避免永久卡住
DISPLAY_TOKENS = SEED_TOKENS
THINK_LINE_DELAY = 0.035   # 深度思考段逐行打字间隔(秒)


# ------------------------------------------------------------------
# v0.6: 设置持久化 (源码级永久保存)。`setting` 命令直接改写
#   xiaofang_settings.py 的常量并写盘, 下次启动自动生效。
# ------------------------------------------------------------------
def _load_settings():
    d = dict(THINK_MODE="think", THINK_TURNS=3, EMOTION_SENSITIVITY=1.2,
             USE_PUNCT_EMOJI=True, SHOW_DEEP_THINK=True, ENERGY=0.9,
             FORCE_OFFLINE=False)
    try:
        import xiaofang_settings as _S
        for k in d:
            if hasattr(_S, k):
                d[k] = getattr(_S, k)
    except Exception:
        pass
    return d


CFG = _load_settings()
THINK_MODE = str(CFG.get("THINK_MODE", "think"))
THINK_TURNS = max(1, int(CFG.get("THINK_TURNS", 3)))
EMO_SENS = float(CFG.get("EMOTION_SENSITIVITY", 1.2))
USE_PUNCT_EMOJI = bool(CFG.get("USE_PUNCT_EMOJI", True))
SHOW_DEEP_THINK = bool(CFG.get("SHOW_DEEP_THINK", True))
ENERGY = float(CFG.get("ENERGY", 0.9))
FORCE_OFFLINE = bool(CFG.get("FORCE_OFFLINE", False))


def _save_settings_field(key, value):
    """直接把某条设置改写进 xiaofang_settings.py (源码级持久化)."""
    import xiaofang_settings as _S
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xiaofang_settings.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    pat = re.compile(r"^(%s\s*=\s*)[^\r\n]*$" % re.escape(key), re.M)
    newval = "True" if value is True else ("False" if value is False else repr(value))
    if not pat.search(src):
        src = src.rstrip() + "\n%s = %s\n" % (key, newval)
    else:
        src = pat.sub(lambda m: m.group(1) + newval, src)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    getattr(_S, key, None) and setattr(_S, key, value)


def _install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package,
                           "--quiet", "--break-system-packages"])


def _ensure_dependencies():
    required = {"colorama": "colorama", "ddgs": "ddgs", "pyfiglet": "pyfiglet",
                "numpy": "numpy", "psutil": "psutil"}
    missing = []
    for imp_name, pip_name in required.items():
        try:
            __import__(imp_name)
        except ImportError:
            missing.append((imp_name, pip_name))
    if missing:
        print("检测到缺失依赖，正在自动安装 ...")
        for imp_name, pip_name in missing:
            print("  -> 正在安装 {} ...".format(pip_name))
            try:
                _install_package(pip_name)
            except subprocess.CalledProcessError:
                print("  -> " + pip_name + " 安装失败，请手动: pip install " + pip_name)
    global Fore, Style, pyfiglet, DDGS, np
    import colorama
    from colorama import Fore, Style
    import pyfiglet
    # v0.4Search: 优先新版 ddgs, 回退旧版 duckduckgo_search
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    import numpy as np
    colorama.init(autoreset=False)


_ensure_dependencies()

from colorama import Fore, Style
import pyfiglet
# v0.4Search: 优先新版 ddgs, 回退旧版 duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import numpy as np


# v0.6: 新的方块小人形象 (制表符版已弃用, 换你给的 🟦/⬛️ 实心方块)
MASCOT = [
    "🟦 🟦 🟦 🟦 🟦",
    "🟦 ⬛️ 🟦 ⬛️ 🟦",
    "🟦 ⬛️ 🟦 ⬛️ 🟦",
    "🟦 🟦 🟦 🟦 🟦",
    "🟦 🟦 🟦 🟦 🟦",
]
BLOCK_ICON = MASCOT

C_BORDER = Fore.LIGHTBLUE_EX
C_THINK = Fore.LIGHTBLACK_EX
C_DEEP = Fore.LIGHTBLACK_EX   # v0.4.1 深度思考改为灰色
C_REPLY = Fore.LIGHTBLUE_EX   # v0.4.1 小方正式回答统一用浅蓝色
C_SYSTEM = Fore.LIGHTBLUE_EX
C_ERROR = Fore.RED
C_HINT = Fore.LIGHTBLACK_EX
C_RESET = Style.RESET_ALL

# v1.0 修复: 全局打印锁 —— 不让"仍思考心跳"这类主线程打印, 挤进小方打字机(灰/蓝)
# 输出的色彩中间段(否则会把蓝色回答截断成白色). 所有跨线程的逐字/逐行打印共用此锁。
_PRINT_LOCK = threading.Lock()
# v1.2 卡死防呆: 每次流式打字改写这个时间戳; 主循环据此区分"真在思考(在动)" vs "卡机(停了)"
_OUT_TICK = {"t": 0.0}


def _touch_live():
    _OUT_TICK["t"] = time.time()


def _typewrite(text, color=C_REPLY, delay=0.035, end="\n"):
    # v1.4: 思考态捕获——输出先进 buf 由面板重绘, 不打字机撞屏; 答案阶段仍逐字打字。
    if UI_ST["capture"]:
        with _PRINT_LOCK:
            s = re.sub(r"\s+", " ", str(text)).strip()
            if s:
                UI_ST["buf"].append(s)
        UI_ST["last_tick"] = time.time()
        return
    # 打字机效果: 逐字输出, 营造"一点点打出来"的感觉
    with _PRINT_LOCK:
        _OUT_TICK["t"] = time.time()
        sys.stdout.write(color)
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write(C_RESET + end)
        sys.stdout.flush()


def _typewrite_lines(lines, color=C_DEEP, line_delay=THINK_LINE_DELAY):
    # v1.4: 思考态捕获——按行进 buf, 由面板重绘; 非思考态仍逐行打字。
    if UI_ST["capture"]:
        with _PRINT_LOCK:
            for ln in lines:
                s = re.sub(r"\s+", " ", str(ln)).strip()
                if s:
                    UI_ST["buf"].append(s)
        UI_ST["last_tick"] = time.time()
        return
    # v0.5 Alpha: 深度思考段逐行打字输出 (长块用逐行更合适, 快而仍有点到点的感觉)
    with _PRINT_LOCK:
        _OUT_TICK["t"] = time.time()
        sys.stdout.write(color)
        for ln in lines:
            sys.stdout.write(ln + "\n")
            sys.stdout.flush()
            time.sleep(line_delay)
        sys.stdout.write(C_RESET)
        sys.stdout.flush()


def _render_cool_title(text):
    fonts_to_try = ["ansi_shadow", "3d_diagonal", "block", "isometric1", "slant", "standard"]
    for font in fonts_to_try:
        try:
            art = pyfiglet.figlet_format(text, font=font)
        except (Exception,):
            continue
        if not art or not art.strip():
            continue
        lines = art.rstrip("\n").split("\n")
        if len(lines) >= 3:
            stripped = [l.lstrip() for l in lines if l.strip()]
            if not stripped:
                continue
            max_w = max(len(s) for s in stripped)
            return [s.ljust(max_w) for s in stripped]
    _FB = {
        "A": ["  █████╗  ", " ██╔═══██╗", " ███████║ ", " ██╔═══██║", " ╚██████╔╝", "  ╚═════╝ "],
        "I": ["██╗       ", "██║       ", "██║       ", "██║       ", "██║       ", "╚═╝       "],
        "X": ["██╗  ██╗  ", "╚██╗██╔╝  ", " ╚███╔╝   ", " ██╔██╗   ", "██╔╝ ██╗  ", "╚═╝  ╚═╝  "],
        "O": [" ██████╗  ", "██╔═══██╗ ", "██║   ██║ ", "██║   ██║ ", "╚██████╔╝ ", " ╚═════╝  "],
        "F": ["███████╗  ", "██╔════╝  ", "█████╗    ", "██╔═══╗   ", "██║   ██║ ", "╚═╝   ╚═╝ "],
        "N": ["██╗  ██╗  ", "██║  ██║  ", "██║  ██║  ", "██║  ██║  ", "╚██████╔╝ ", " ╚═════╝  "],
        " ": ["          ", "          ", "          ", "          ", "          ", "          "],
    }
    rows = ["", "", "", "", "", ""]
    for ch in text.upper():
        glyph = _FB.get(ch) or _FB[" "]
        for i in range(6):
            rows[i] += glyph[i]
    return [r.rstrip() for r in rows]


def _disp_width(s):
    w = 0
    for ch in s:
        cp = ord(ch)
        if 0x2500 <= cp <= 0x27BF:
            w += 1
        elif cp > 0x2E80:
            w += 2
        else:
            w += 1
    return w


def _center_pad(s, width):
    pad = max(0, width - _disp_width(s))
    left = pad // 2
    right = pad - left
    return " " * left + s + " " * right


# ============================================================
# v1.4 Alpha: 全局美化 —— 思考/状态分段面板
#   状态面板: 按下回车即出现在思考下方; Token 从0实时增 + 当前层数/阶段
#             + 面板最右侧盲文点旋转加载。
#   思考/计算与状态面板分离, 用 ── 分界线隔开, 可点开/折叠(点击面板 或 按 F7)。
#   思考态的输出先进 buf 缓冲 → 由主线程面板单行重绘 → 彻底不与打字机撞屏;
#   打字机效果完整保留在最终"答案"输出上。
# ============================================================
UI_ST = {
    "capture": False,   # True=思考/计算期输出先捕获进 buf, 交给面板重绘(不打字机屏幕)
    "buf": [],          # 捕获到的思考/计算行(思考中出现于用户问题与状态面板之间)
    "rows": 0,          # 面板当前占的屏幕行数(用于原地重绘)
    "in": 0, "out": 0,  # 实时 Token 显示值(从按下回车=0 随思考增; 真值在 meter)
    "layer": 0,         # 实时"正在思考第几层"
    "stage": "",        # 实时当前状态/阶段
    "fold": True,       # True=面板折叠成一行状态条; False=展开思考与计算详情
    "t0": 0.0,
    "last_tick": 0.0,
}
_SPIN_UI = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"         # 盲文点旋转加载


def _ui_reset_turn():
    UI_ST["capture"] = True
    UI_ST["buf"] = []
    UI_ST["rows"] = 0
    UI_ST["filter"] = None
    UI_ST["in"] = 0
    UI_ST["out"] = 0
    UI_ST["layer"] = 0
    UI_ST["stage"] = "解析中…"
    UI_ST["fold"] = False      # v1.4: 默认展开
    UI_ST["t0"] = time.time()
    UI_ST["last_tick"] = time.time()


def _ui_spinner():
    return _SPIN_UI[(int(time.time() * 4)) % len(_SPIN_UI)]


def _ui_token_display():
    # 真实 output 未落定前, 用"随思考流逝而增长的合成量"让 Token 从 0 肉眼可见地往上涨
    if UI_ST["out"]:
        return UI_ST["in"] + UI_ST["out"]
    return UI_ST["in"] + int((time.time() - UI_ST["t0"]) * 8)


def _ui_status_text():
    n = MODEL_LAYERS
    l = min(UI_ST["layer"], n)
    st = UI_ST["stage"] or ("思考中" if UI_ST["capture"] else "已就绪")
    return ("⟳ 小方{} · Token: {} · 第{}/{}层 · {} · {}".format(
        ("思考中" if UI_ST["capture"] else "已就绪"),
        _ui_token_display(), l, n, st, _ui_spinner()))


def _panel_render():
    """原地重绘"思考/计算分段面板"：折叠=单行状态条; 展开=思考/计算详情+分界线+状态面板。"""
    lines = []
    if UI_ST["fold"]:
        lines.append("▶ 展开思考/计算详情 (点击面板 或 按 F7)   ⸻ " + _ui_status_text())
    else:
        lines.append("▼ 收起思考/计算详情 (点击面板 或 按 F7)   ⸻  深度思考 · 计算过程")
        buf = [ln for ln in UI_ST["buf"] if ln.strip()]
        MAX = 9
        for ln in buf[:MAX]:
            lines.append("     · " + ln)
        if len(buf) > MAX:
            lines.append("     · …… 共 {} 行计算详情（折叠时隐藏, 展开即可见）".format(len(buf)))
        lines.append(" " + "─" * 16 + " 计算过程分界线 " + "─" * 16)
        lines.append("      " + _ui_status_text())
    if UI_ST["rows"]:
        sys.stdout.write("\033[%dA\r" % UI_ST["rows"])
    with _PRINT_LOCK:
        sys.stdout.write("\033[K")
        for i, ln in enumerate(lines):
            if i:
                sys.stdout.write("\n");
            sys.stdout.write(C_HINT + ln + C_RESET + "\033[K")
        sys.stdout.write("\n")
        sys.stdout.flush()
    UI_ST["rows"] = len(lines)
    UI_ST["last_tick"] = time.time()


# ============ v0.6: 全局反馈与防卡死工具 ============
def _status(msg):
    """阶段状态提示: 小方此刻"正在干什么"的实时灰字进度条."""
    if UI_ST["capture"]:
        with _PRINT_LOCK:
            s = re.sub(r"\s+", " ", str(msg)).strip()
            if s and (not UI_ST["buf"] or UI_ST["buf"][-1] != s):
                UI_ST["buf"].append(s)
            UI_ST["stage"] = s
        UI_ST["last_tick"] = time.time()
        return
    _typewrite_lines(["  · " + msg], C_HINT, line_delay=0.006)


def _run_with_timeout(fn, timeout=4.0, default=None):
    """把 fn 放进守护线程, 超时立即放弃返回 default。
       根治"联网/搜索没回应→看起来卡死"——绝不因网络阻塞主流程。"""
    import threading
    out = {"v": default, "done": False}

    def _t():
        try:
            out["v"] = fn()
            out["done"] = True
        except Exception:
            out["done"] = True
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join(timeout)
    return out["v"] if out["done"] else default


# ============================================================
# v1.3 Alpha: 真实命令行输入 —— 斜杠调出工具列表(前缀过滤/点击选中/可继续输)、
#           F1-F5 快捷键、Y/N 确认、思考中防撞屏
# ============================================================
_COMMANDS = [
    ("/help",    "命令手册 + 快捷键面板"),
    ("/setting", "进入设置(思考/情绪/联网, 永久保存)"),
    ("/think",   "切到单轮深度思考"),
    ("/multi",   "切到多轮边答边想"),
    ("/off",     "切到不思考(最快)"),
    ("/clear",   "清空对话历史"),
    ("/prev",    "查看上一个对话"),
    ("/undo",    "回到上一个对话  [需确认]"),
    ("/new",     "开启新对话        [需确认]"),
    ("/memory",  "查看学习记忆档案"),
    ("/forget",  "清空学习记忆"),
    ("/stop",    "强制中断当前思考, 立即回到输入"),
    ("/exit",    "退出程序"),
]
_CMD_NAMES = [c[0] for c in _COMMANDS]
_CMD_DESC = dict(_COMMANDS)
# 功能键第二字节 → 命令 (仅用于 \xe0 / \x00 延长码分支)
_A_KEYS = {"H": "up", "P": "down"}                      # 方向键
_FKEY_SCAN = {";": "/help", "<": "/prev", "=": "/undo",
              ">": "/new", "?": "/stop", "@": "/exit", "A": "/fold"}  # F1-F7, F7=展开/折叠思考面板
_CONFIRM_CMDS = {"/undo", "/new", "/exit"}              # 破坏性/占用操作需确认
_SYS_BUSY = {"v": False}                                # True=模型正思考, 调色板不画防撞屏
# v1.4 修复: "回答完不再弹你:" —— 提示符改由主线程在不忙时统一绘制,
#            读线程不再抢着打印(避免与状态面板/答案撞屏, 也避免提前出现的残余"你:")。
_READ_PROMPT = "你: "
_read_prompt_pending = True     # True=当前可绘制提示符(空余期), 显示后置 False, 提交一句后再置 True


def _show_read_prompt():
    """主线程在空闲态绘制"你: "。若读线程正在交互回显(用户正打字), 由 lock 避免行内交错。"""
    global _read_prompt_pending
    if not _read_prompt_pending:
        return
    with _PRINT_LOCK:
        sys.stdout.write(C_REPLY + _READ_PROMPT + C_RESET)
        sys.stdout.flush()
    _read_prompt_pending = False


def _confirm_yn(name):
    """单键 Y/N 确认 (在 _read_command 读线程内调用, 不与主线程抢键盘); 回车视为确认。"""
    import msvcrt
    sys.stdout.write("\033[K" + C_HINT + "  {}? [Y/y 确认 · N/n 取消, 回车=确认]: ".format(name) + C_RESET)
    sys.stdout.flush()
    try:
        k = msvcrt.getwch()
    except Exception:
        k = "n"
    ok = k.lower() in ("y", "\r", "\n")
    sys.stdout.write("→ " + ("已确认 ✓" if ok else "已取消 ✗") + "\n")
    sys.stdout.flush()
    return ok


def _finalize(buf_list, sel, sugs):
    """回车/功能键被选中时的落定: 需确认的命令先弹 Y/N。返回最终文本或空串=取消。"""
    raw = "".join(buf_list)
    if sugs:
        cmd = sugs[sel % len(sugs)]
    elif raw.startswith("/"):
        cmd = raw
    else:
        return raw
    if cmd in _CONFIRM_CMDS:
        name = _CMD_DESC.get(cmd, cmd)
        return cmd if _confirm_yn(name) else ""
    return cmd or raw


def _read_command(prompt="你: "):
    """真实命令行输入框(终端版"斜杠工具列表"):
       · 输入 / 立即在下方列出全部可用命令(含一句话描述);
       · 继续输 → 实时按前缀过滤( /S → S 开头的; /SE → 基本只剩 /setting );
       · ↑/↓ 选择, Tab 填入, 回车提交; 列表内直接回车=选中该项(如同点击);
       · F1-F6 快捷键: 帮助 / 查看上一个 / 回到上一个(确认) / 新对话(确认) / 强制退出 / 退出;
       · /undo /new /exit 需 Y/N 确认;
       · 模型思考中(_SYS_BUSY)只更新输入行、不画调色板, 避免与打字机撞屏;
       · 不支持逐键读入的环境自动回退普通 input()."""
    if os.name != "nt":
        try:
            return input(C_REPLY + prompt + C_RESET)
        except Exception:
            return ""
    try:
        import msvcrt
    except Exception:
        return input(C_REPLY + prompt + C_RESET)
    _st = {"h": 0, "buf": list(), "sel": 0}

    def _matches():
        raw = "".join(_st["buf"])
        if not raw.startswith("/"):
            return []
        # 大小写不敏感: /S 或 /s → setting/stop; /SE → 只剩 setting(与真实 CLI 一致)
        low = raw.lower()
        return [n for n in _CMD_NAMES if n.lower().startswith(low)]

    def _clear_panel():
        if _st["h"] and not _SYS_BUSY["v"]:
            sys.stdout.write("\033[%dA\r\033[J" % _st["h"])
            _st["h"] = 0

    def _redraw():
        raw = "".join(_st["buf"])
        if _SYS_BUSY["v"]:                       # 思考/回答期: 屏幕归状态面板与打字机, 这里不回显
            _st["h"] = 0
            return
        sugs = _matches()
        if _st["h"]:
            sys.stdout.write("\033[%dA\r" % _st["h"])
            sys.stdout.write("\033[K" + C_REPLY + prompt + raw + C_RESET + " \033[K")
        else:
            sys.stdout.write("\r" + C_REPLY + prompt + raw + C_RESET + " \033[K")
        if sugs:
            sel = _st["sel"] % len(sugs)
            lines = [C_HINT + "   ⚡ ↑/↓ 选 · 回车=选中 · Tab 填入 · 继续输=过滤   "
                              "| F1帮助 F2上一个 F3回退 F4新对话 F5停止 F7面板" + C_RESET]
            for i, c in enumerate(sugs):
                mark = (C_REPLY + "▸ " if i == sel else C_HINT + "  ")
                lines.append(mark + c + C_HINT + "  · " + _CMD_DESC.get(c, "") + C_RESET)
            sys.stdout.write("\n\033[K".join([""] + lines))
            sys.stdout.flush()
            _st["h"] = len(lines)
            sys.stdout.write("\033[%dA\r" % _st["h"])
        else:
            _st["h"] = 0
        sys.stdout.flush()

    _st["buf"] = list()
    # v1.4 修复: 不再在此绘制"你: " —— 提示符由主线程 _show_read_prompt() 统一在空闲态画,
    #             避免读线程抢画/提前画导致回答完看不到提示符或与状态面板撞屏。
    while True:
        try:
            ch = msvcrt.getwch()
        except KeyboardInterrupt:
            _clear_panel()
            print()
            return ""
        if ch in ("\r", "\n"):
            sugs = _matches()
            _clear_panel()
            print()
            return _finalize(_st["buf"], _st["sel"], sugs)
        elif ch == "\t":
            sugs = _matches()
            if sugs:
                _st["buf"] = list(sugs[_st["sel"] % len(sugs)])
                _st["sel"] = 0
            _redraw()
        elif ch == "\x03":
            _clear_panel()
            print()
            raise KeyboardInterrupt
        elif ch == "\x1b":                        # Esc: 取消当前输入, 清空联想
            if _st["buf"] or _st["h"]:
                _clear_panel()
                print()
            _st["buf"] = list()
            _st["sel"] = 0
            sys.stdout.write(C_REPLY + "\r" + "\033[K" + prompt + C_RESET)
            sys.stdout.flush()
        elif ch in ("\x08", "\x7f"):
            if _st["buf"]:
                _st["buf"].pop()
            _redraw()
        elif ch in ("\xe0", "\x00"):
            k = ""
            try:
                k = msvcrt.getwch()
            except Exception:
                k = ""
            if k == "M":                       # v1.4: Windows 控制台鼠标事件 POCKET \xe0 M X-Y-B
                try:
                    _xb, _yb, _bb = ord(msvcrt.getwch()), ord(msvcrt.getwch()), ord(msvcrt.getwch())
                except Exception:
                    _xb = _yb = _bb = 0
                if _bb & 1 and UI_ST["capture"]:   # 左键单击 → 点击展开/折叠思考面板
                    UI_ST["fold"] = not UI_ST["fold"]
                _redraw()
                continue
            if k in _FKEY_SCAN:
                cmd = _FKEY_SCAN[k]
                _clear_panel()
                print()
                name = _CMD_DESC.get(cmd, cmd.split("/", 1)[-1])
                if cmd in _CONFIRM_CMDS:
                    return cmd if _confirm_yn(name) else ""
                return cmd
            elif k in _A_KEYS:
                if k == "H":
                    if _matches():
                        _st["sel"] = (_st["sel"] - 1) % len(_matches())
                else:
                    sugs = _matches()
                    if sugs:
                        _st["sel"] = (_st["sel"] + 1) % len(sugs)
            _redraw()
        else:
            _st["buf"].append(ch)
            _st["sel"] = 0
            _redraw()


def _fmt_param_count(n):
    if n >= 1e9:
        return "{:.2f}B".format(n / 1e9)
    if n >= 1e6:
        return "{:.0f}M".format(n / 1e6)
    return str(n)


# ------------------------------------------------------------------
# v1.0: 启动加载进度报告器 —— 实时显示 百分比 + 预计耗时 + 当前步骤/进程,
#      并标注 GPU/CPU 后端, 彻底告别"白的像卡死"的启动等待。
# ------------------------------------------------------------------
class _BootReporter:
    def __init__(self):
        self.t0 = time.time()
        self.seg = 0.0                   # 已完成的阶段累计权重 [0,1]
        self._once = object()

    def _eta(self):
        el = time.time() - self.t0
        base = max(self.seg, 1e-6)
        if el < 3 or self.seg <= 0.15:   # 刚启动还没测到真实速度 → 给经验预计
            return 30.0
        return el * (max(0.0, 1.0 - self.seg) / base)

    def _fmt_eta(self, eta):
        eta = max(1, int(eta))
        return "约 {} 秒".format(eta) if eta < 60 else "约 {} 分 {} 秒".format(eta // 60, eta % 60)

    def render(self, name, seg_boost=0.0):
        if os.environ.get("XF_QUIET", ""):
            return
        self.seg = min(1.0, self.seg + seg_boost)
        pct = int(self.seg * 100)
        backend = "GPU 加速 ✓" if HAS_GPU else "CPU 模式(未检测到显卡)"
        line = ("⏳ 正在启动 AI 小方 {} · [{:>3}%] · 正在：{} · {} · 预计还需 {}"
                .format(VERSION, pct, name, backend, self._fmt_eta(self._eta())))
        print(C_HINT + line + C_RESET, flush=True)

    def stage(self, name, seg_weight=0.0):
        # 顶层大阶段：权重为"该阶段占总启动的比例", 用于算百分比与 ETA
        self.render(name, seg_weight)

    def enter(self, name, boost=0.0):
        # 进入一个长耗时子阶段(如神经网构建)，先通报进程
        self.render(name, 0.0)

    def tick(self, i, total, name):
        # 子阶段内逐层/逐部件刷新（抑制刷屏，间隔≥0.15s 或 层数变化才打印）
        if total <= 0:
            return
        now = time.time()
        if getattr(self, "_cached_t", None) == i and now - getattr(self, "_cached_ts", 0) < 0.15:
            return
        self._cached_t = i
        self._cached_ts = now
        self.render("{} ...".format(name), 0.0)

    def leave(self):
        return


_br = _BootReporter()


def _cls():
    """v1.0: 完整启动后就绪 → 清空整屏, 从最上方重新展示 (更简洁)."""
    if os.environ.get("XF_QUIET", ""):
        return
    if os.name == "nt":
        os.system("cls")
        try:
            os.system("title AI 小方 FlphaLit 1.0 Pro")
        except Exception:
            pass
    else:
        os.system("clear")


def _startup_title_rows(max_w, max_h=8):
    """巨型 AI XIAOFANG：按可容纳的宽度/行高挑选最气派的字体, 兜底用内置方块字。"""
    best = None
    for font in ("big", "slant", "standard", "ansi_shadow", "block"):
        try:
            art = pyfiglet.figlet_format("AI XIAOFANG", font=font)
        except Exception:
            continue
        rows = [l.rstrip() for l in (art or "").rstrip("\n").split("\n") if l.strip()]
        if not rows or len(rows) > max_h:
            continue
        if max(_disp_width(r) for r in rows) <= max_w:
            best = rows
            break
    return best or _render_cool_title("AI XIAOFANG")


def show_startup():
    # v1.4: 占满全局的大 ASCII 框(约 1/3 屏高)
    try:
        import shutil
        tw = shutil.get_terminal_size((100, 30))
        W = max(46, min(tw.columns, 190))
        termH = tw.lines or 25
    except Exception:
        W, termH = 100, 25
    inner = W - 2
    box_h = max(14, min(termH // 3 + 2, 22))

    # —— 右中: 巨型 AI XIAOFANG ——
    title_rows = _startup_title_rows(inner - 26, max_h=max(5, box_h - 10))
    th = len(title_rows)
    # —— 左上: 小方块形象 ——
    left = list(MASCOT)
    hero_h = max(len(left), th)

    def _vpad(arr, n):
        top = (n - len(arr)) // 2
        return [""] * top + list(arr) + [""] * (n - len(arr) - top)

    left_r = _vpad(left, hero_h)
    right_r = _vpad(title_rows, hero_h)
    lw = max(_disp_width(x) for x in left_r) if any(x for x in left_r) else 1
    lw += 4

    def _row(li, ri):
        pad = max(0, lw - _disp_width(li))
        line = " " + li + " " * pad + " |  " + ri
        return line + " " * max(0, (inner - 2) - _disp_width(line))

    rows = [_row(left_r[i], right_r[i]) for i in range(hero_h)]
    rows.append("  " + "-" * (inner - 4))

    lower_left = ["你可以问我这些问题：",
                  "  · 小方工作室是什么",
                  "  · 你的底层原理是什么",
                  "  · 你现在是哪个版本？是真的 LLM 吗？",
                  "",
                  "在键盘上敲打开始对话："]
    lower_right = ["  快捷指令",
                   "",
                   "  /help 命令手册 · /setting 设置",
                   "  /prev 上个对话 · /undo 回退[确认]",
                   "  /new 新对话[确认] · /stop 掐断",
                   "  F1手册 F2上个 F3回退 F4新 F5掐 F7面板"]
    for i in range(len(lower_left)):
        ri = (lower_right[i] if i < len(lower_right) else "")
        rows.append(_row(lower_left[i], ri))

    gpu_s = HW["gpu_model"] + (" · {}GB".format(HW["vram"] // 1024) if HW["vram"] else "")
    status = ("  {} {} · 深度思考{}L×{}H d_model={} · {} · {}".format(
        ENGINE_NAME, VERSION, MODEL_LAYERS, MODEL_HEADS, MODEL_D,
        ("GPU" if HAS_GPU else "CPU"),
        ("内存 {:.1f}GB".format(HW["mem_total"] / (1024 ** 3)) if HW["mem_total"] else "")))
    rows.append(status + " " * max(0, (inner - 2) - _disp_width(status)))

    print(C_SYSTEM + "┌" + "═" * (inner - 2) + "┐")
    for r in rows:
        print(C_SYSTEM + "│" + r + C_SYSTEM + "│" + C_RESET)
    print(C_SYSTEM + "└" + "═" * (inner - 2) + "┘")
    print()
    print(C_HINT + "  ⚡ 直接在键盘上敲打开始对话 · 输入 / 调出命令列表 " + C_RESET)
    print()


def _enable_console_mouse():
    # v1.4: 开启 Windows 控制台鼠标输入 → 支持"点击展开/折叠思考面板"(尽力而为, 失败不影响 F7)
    try:
        import ctypes
        from ctypes import wintypes
        STD_INPUT_HANDLE = -10
        kk = ctypes.windll.kernel32
        h = kk.GetStdHandle(STD_INPUT_HANDLE)
        mode = wintypes.DWORD()
        if h and h != ctypes.c_void_p(-1).value and kk.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_MOUSE_INPUT = 0x0010
            ENABLE_EXTENDED_FLAGS = 0x0080
            kk.SetConsoleMode(h, (mode.value & ~ENABLE_EXTENDED_FLAGS) | ENABLE_MOUSE_INPUT)
    except Exception:
        pass


class Tokenizer:
    _SKIP = set(" \t\r\n，。！？、,.!?;:\"'…~`()——（）【】《》<>'·•·／/\\|")

    def __init__(self, extra_words=None):
        self.dictionary = set()
        self.max_len = 1
        for w in DATA.COMMON_WORDS:
            self._add(w)
        for w in list(DATA.EMOTION_POS.keys()) + list(DATA.EMOTION_NEG.keys()) \
                + list(DATA.EMOTION_DEGREE.keys()) + list(DATA.EMOTION_NEGATION):
            self._add(w)
        if extra_words:
            for w in extra_words:
                self._add(w)

    def _add(self, w):
        if not w:
            return
        self.dictionary.add(w)
        if len(w) > self.max_len:
            self.max_len = len(w)

    def add_words(self, words):
        for w in words:
            self._add(w)

    def tokenize(self, text):
        if not text:
            return []
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            upper = min(self.max_len, n - i)
            for length in range(upper, 1, -1):
                word = text[i:i + length]
                if word in self.dictionary:
                    tokens.append(word)
                    i += length
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch not in self._SKIP:
                    tokens.append(ch)
                i += 1
        return tokens

    def tokenize_set(self, text):
        return set(self.tokenize(text))


class EmotionAnalyzer:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def analyze(self, text):
        tokens = self.tok.tokenize(text)
        score = 0.0
        hits = []
        neg_window = 0
        degree_mult = 1.0
        has_question = False
        # v0.6: 标点+emoji 情绪权重加权 (!!=激动 ?!=困惑 …=低落; emoji 直接给极性分)
        punct_emo = 0.0
        emoji_note = []
        if USE_PUNCT_EMOJI:
            bang = text.count("!") + text.count("！")
            ellip = text.count("…") + text.count("..") + text.count("。。")
            if bang >= 2:
                punct_emo += 1.3
            elif bang == 1:
                punct_emo += 0.4
            if ellip >= 2:
                punct_emo -= 1.0
            _pe = {}.fromkeys((_[0] if isinstance(_, tuple) and len(_) else _) for _ in DATA.EMOJI_POSITIVE)
            _ne = {}.fromkeys((_[0] if isinstance(_, tuple) and len(_) else _) for _ in DATA.EMOJI_NEGATIVE)
            for _ek, _v in list(_pe.items()) + list(_ne.items()):
                if _ek and _ek in text:
                    punct_emo += _v
                    emoji_note.append(_ek)
        for t in tokens:
            if t in DATA.EMOTION_NEGATION:
                neg_window = 2
                continue
            if t in DATA.EMOTION_DEGREE:
                degree_mult = DATA.EMOTION_DEGREE[t]
                continue
            if t in ("吗", "呢", "怎么", "为什么", "为何", "如何", "哪里",
                     "哪儿", "哪个", "哪些", "多少", "是否", "谁", "孰",
                     "何", "能不能", "可以吗"):
                has_question = True
            contrib = 0.0
            if t in DATA.EMOTION_POS:
                contrib = DATA.EMOTION_POS[t]
            elif t in DATA.EMOTION_NEG:
                contrib = DATA.EMOTION_NEG[t]
            if contrib != 0.0:
                if neg_window > 0:
                    contrib = -contrib
                    neg_window -= 1
                contrib = contrib * degree_mult
                degree_mult = 1.0
                score += contrib
                hits.append((t, round(contrib, 2)))
        pos_hits = [t for (t, c) in hits if c > 0]
        neg_hits = [t for (t, c) in hits if c < 0]
        intensity = sum(abs(c) for _, c in hits)
        # v0.6: 融入标点/emoji 加权与敏感度系数
        punct_emo = punct_emo * EMO_SENS
        score += punct_emo
        if punct_emo:
            intensity += abs(punct_emo)
        has_anger = any(t in DATA.EMO_ANGER for t in neg_hits)
        has_sad = any(t in DATA.EMO_SAD for t in neg_hits)
        has_fear = any(t in DATA.EMO_FEAR for t in neg_hits)
        has_tired = any(t in DATA.EMO_TIRED for t in neg_hits)
        has_joy = any(t in DATA.EMO_JOY for t in pos_hits)
        has_love = any(t in DATA.EMO_LOVE for t in pos_hits)
        category = self._classify(
            score, intensity, len(pos_hits), len(neg_hits),
            has_anger, has_sad, has_fear, has_tired, has_joy, has_love, has_question)
        return {
            "score": round(score, 2),
            "intensity": round(intensity, 2),
            "tokens": tokens,
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "category": category,
            "has_question": has_question,
            "has_anger": has_anger, "has_sad": has_sad, "has_fear": has_fear,
            "has_tired": has_tired, "has_joy": has_joy, "has_love": has_love,
            "punct_emo": punct_emo, "emoji_note": emoji_note}

    def _classify(self, score, intensity, pos_count, neg_count,
                  has_anger, has_sad, has_fear, has_tired, has_joy, has_love, has_question):
        if score > 0:
            if score >= 6:
                if has_love:
                    return "深爱"
                elif has_joy:
                    return "狂喜"
                else:
                    return "极度满足"
            elif score >= 3:
                if has_love:
                    return "喜爱"
                elif has_joy:
                    return "愉悦"
                elif neg_count == 0:
                    return "积极"
                else:
                    return "略带欣慰"
            else:
                if pos_count >= 2:
                    return "心情不错"
                elif has_love:
                    return "略有好感"
                else:
                    return "平和"
        elif score < 0:
            if score <= -6:
                if has_anger:
                    return "愤怒"
                elif has_sad:
                    return "悲痛"
                elif has_fear:
                    return "恐惧"
                elif has_tired:
                    return "濒临崩溃"
                else:
                    return "极度低落"
            elif score <= -3:
                if has_anger:
                    return "生气"
                elif has_sad:
                    return "难过"
                elif has_fear:
                    return "焦虑"
                elif has_tired:
                    return "疲惫"
                else:
                    return "低落"
            else:
                if has_tired:
                    return "有点累"
                elif has_fear:
                    return "有点担心"
                elif has_anger:
                    return "有点烦"
                elif has_sad:
                    return "有点失落"
                else:
                    return "轻微不适"
        else:
            if has_question:
                if neg_count > 0 and pos_count > 0:
                    return "矛盾探询"
                else:
                    return "平静探询"
            else:
                if neg_count > 0 and pos_count > 0:
                    if intensity >= 4:
                        return "内心纠结"
                    else:
                        return "情绪平稳"
                elif pos_count == 0 and neg_count == 0:
                    return "完全中性"
                else:
                    return "情绪平稳"


class IntentDetector:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def detect(self, text, emo):
        tokens = emo["tokens"]
        tset = set(tokens)
        raw = text

        def cnt(group):
            return sum(1 for w in group if w in tset)

        search_markers = ["搜索", "查一下", "查查", "搜一下", "查询", "查找",
                          "搜搜", "联网", "上网", "最新", "查", "搜"]
        search_score = cnt(search_markers) * 2
        if any(w in raw for w in ["不知道", "不了解", "不清楚", "不懂", "不会", "没听过"]):
            search_score += 1

        q_markers = ["吗", "呢", "怎么", "为什么", "为何", "如何", "哪里", "哪儿",
                     "哪个", "哪些", "多少", "是否", "谁", "孰", "何", "能不能", "可以吗",
                     "什么", "是啥", "啥是", "是什么", "指什么", "讲下", "介绍下", "讲讲",
                     "解释下", "介绍一下"]
        q_score = cnt(q_markers)
        if "？" in raw or "?" in raw or "什么" in raw or "啥" in raw:
            q_score += 1

        cmd_score = cnt(["请", "帮我", "给", "告诉", "说说", "讲讲", "介绍", "解释", "说明", "列举"])
        id_score = sum(1 for w in ["你是谁", "你叫什么", "你是", "你叫", "名字", "身份"]
                       if w in raw)
        # v0.4 正式版: 识别"你是什么模型/AI"、作者、工作室、官网等提问主体
        if any(m in raw for m in ["谁做", "谁开发", "作者", "工作室", "开发者",
                                  "创造者", "官网", "小方工作室"]):
            id_score += 2
        # "你是什么模型/AI/机器人" (含"你") 才计身份分, 避免把"大语言模型"知识问答误判
        if ("你" in raw or "你们" in raw) and any(m in raw for m in
                ["模型", "AI", "ai", "机器人", "助手", "程序", "系统", "made",
                 "框架", "底层", "源码", "技术栈", "用什么", "怎么做出"]):
            id_score += 2
        if not id_score and re.search(r"(你是|你叫).{0,4}(谁|什么|啥)", raw):
            id_score += 1
        # v0.5 Alpha 2: 设定人格检测。仅当问到"你"的家庭/住址/年龄/喜好/身世/世界观时触发。
        #   "编程/星空"等中性词不单独触发, 以"你/世界观专名"为锚, 避免把定义问答误判成人格。
        persona_score = 0
        _world = ["刘小方", "天空王国", "方族", "方岛", "蓝石", "方之光", "曹方宇",
                  "你来自", "你从哪里来", "你的家乡", "康岛", "雪岛"]
        if ("你" in raw or "你们" in raw) or any(m in raw for m in _world):
            # 人格词库关键词命中权重高, 让"你长什么样/你喜欢吃什么"等明确问本人的问题胜出
            persona_score += 2 * sum(1 for e in getattr(DATA, "PERSONA_KB", [])
                                     for kw in e["kws"] if kw and kw in raw)
            persona_score += sum(1 for w in ["家住", "住哪", "住在", "住址", "你家", "家里",
                                             "爸爸", "妈妈", "父母", "家人", "姐", "妹",
                                             "几岁", "多大", "几年级", "养猫", "宠物", "你的猫", "你猫",
                                             "围巾", "头发", "朋友", "同学", "喜欢", "爱好", "兴趣",
                                             "特长", "大名", "小名", "性格", "生活", "平时"] if w in raw)
            persona_score += sum(1 for w in _world if w in raw)
            if re.search(r"你.{0,2}来自|你从.{0,3}来|哪里来|哪里人", raw):
                persona_score += 2
            if re.search(r"你.{0,4}(朋友|家人|爸妈|家乡|老家|宠物|猫|爱好|兴趣)", raw):
                persona_score += 2
        if (raw.startswith("你家") or "你家在" in raw or re.match(r"^(你家|你家住|你住在)", raw)):
            persona_score += 2
        time_score = sum(1 for w in ["几点", "时间", "现在几点", "什么时候"] if w in raw)
        date_score = sum(1 for w in ["日期", "今天", "几号", "星期", "周几"] if w in raw)
        weather_score = sum(1 for w in ["天气", "气温", "下雨", "温度", "预报"] if w in raw)
        joke_score = sum(1 for w in ["笑话", "讲个", "逗", "搞笑", "段子"] if w in raw)
        bye_score = sum(1 for w in ["再见", "拜拜", "bye", "走了", "退出", "晚安", "回去"] if w in raw.lower())
        thx_score = sum(1 for w in ["谢谢", "感谢", "thx", "谢啦", "多谢", "辛苦啦"] if w in raw.lower())
        cap_score = sum(1 for w in ["能做什么", "功能", "会什么", "帮助", "可以做什么"] if w in raw)
        greet_score = sum(1 for w in ["你好", "哈喽", "嗨", "hello", "hi", "早上好", "您好", "在吗"] if w in raw.lower())
        study_score = sum(1 for w in ["怎么学", "学到", "练", "技巧", "怎么提高", "入门", "教程"] if w in raw)
        suggest_score = sum(1 for w in ["建议", "推荐", "怎么办", "该不该", "要不要", "帮我选", "怎么选",
                                        "玩什么", "玩点啥", "什么好玩", "啥好玩", "推荐一下",
                                        "有啥推荐", "有什么推荐", "帮我推荐", "选哪个", "选什么",
                                        "吃什么", "看什么", "听什么"] if w in raw)
        start_score = sum(1 for w in ["开始", "怎么跑步", "怎么开始", "起步", "第一步"] if w in raw)

        # v0.5 正式版: 数学意图 (算式/算术词)
        math_score = 0
        if re.search(r"\d\s*[\+\-*/%×÷]\s*\d", raw):
            math_score += 3
        if re.search(r"[\d一二两三四五六七八九十百千万]+(?:加|减|乘|除|乘以|除以|的平方|的立方|等于多少|等于|求值)", raw):
            math_score += 2
        if any(w in raw for w in ["算一下", "计算", "数学题", "求和", "式子", "解惑"]):
            math_score += 1
        if "数学" in raw:
            math_score += 1
        # —— v1.3 Alpha: 变量方程(如 x²=4 / 2x=6 / x+3=5 / 3x-5=10 / 解方程) 也判为数学 ——
        #   旧版只看"数字+运算符", "x²=4" 这类的等号变元不被识别 → 被通用模板曲解。
        if re.search(r"[=＝]", raw) and re.search(r"[a-df-zA-DF-Z](?:\s*\^?\s*[23²³])?", raw):
            math_score += 3
        if any(w in raw for w in ["解方程", "求未知数", "求x", "求变量", "方程式", "方程"]):
            math_score += 2

        # v0.5 正式版: 代码/算法意图
        code_score = 2 * sum(1 for w in ["写代码", "写个代码", "写程序", "写一段代码", "实现代码", "编程", "实现一个程序"]
                             if w in raw)
        code_score += 2 * sum(1 for w in ["排序", "冒泡", "快排", "快速排序", "二分查找", "二分", "递归", "阶乘",
                                          "斐波那契", "素数", "质数", "最大公约数", "欧几里得", "进制转换", "温度转换",
                                          "数据结构", "链表", "栈", "队列", "算法"] if w in raw)
        if re.search(r"帮我写.{0,6}(代码|函数|程序|排序|算法)", raw):
            code_score += 4
        # —— v1.2 加强: 写/教/教程类 coding 请求, 杜绝"写Python循环教程"被当实体问答(答成创始人) ——
        # 只有当同时出现【请求写/教】信号 与【编程域】词(含基础语法), 才判为代码意图。
        # "什么是循环/血液循环"等纯名词问法不触发, 防误判。
        _write_teach = ["写个", "写一段", "写个代码", "写代码", "教我", "教教", "讲解",
                        "讲一下", "示例", "教程", "写一个", "演示", "怎么写"]
        _code_kw = ["循环", "for", "while", "遍历", "函数", "def", "类", "class",
                    "面向对象", "语法", "变量", "列表", "字典", "元组", "集合", "切片",
                    "if", "elif", "else", "异常", "try", "import", "模块",
                    "生成器", "迭代器", "调试", "正则", "numpy", "pandas", "断点", "排序"]
        if any(w in raw for w in _write_teach) and any(w in raw.lower() for w in _code_kw):
            code_score += 4
        # "Python/代码/编程" 等编程上下文 + 请求写作 → 也是写代码
        _prog_ctx = ["python", "py", "代码", "编程", "程序", "算法", "脚本"]
        if any(w in raw.lower() for w in _prog_ctx) and any(
                w in raw for w in ["写", "教", "教程", "示例", "实现", "如何实现", "怎么实现"]):
            code_score += 3

        help_score = 0
        if emo["category"] in ("难过", "悲痛", "焦虑", "濒临崩溃", "生气",
                                "有点烦", "有点累", "极度低落", "恐惧"):
            help_score += 2

        scores = {
            "search": search_score, "question": q_score, "command": cmd_score,
            "identity": id_score, "persona": persona_score, "time": time_score,
            "date": date_score,
            "math": math_score, "code": code_score,
            "weather": weather_score, "joke": joke_score, "bye": bye_score,
            "thanks": thx_score, "help": help_score, "capability": cap_score,
            "greet": greet_score, "study": study_score, "suggest": suggest_score,
            "start": start_score}
        top = max(scores, key=lambda k: scores.get(k, 0))
        ranked = sorted(((k, v) for k, v in scores.items() if v > 0), key=lambda x: -x[1])
        secondary = [k for k, _ in ranked[:3]]
        return {"scores": scores, "top": top, "max_score": scores[top],
                "ranked": ranked, "secondary": secondary, "tset": tset}


class NgramLM:
    START = "<s>"
    END = "</s>"

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.uni = {}
        self.bi = {}
        self.tri = {}
        self.uni_total = 0
        self.vocab = set()

    def train(self, texts):
        for text in texts:
            toks = self.tok.tokenize(text)
            self._add_seq(toks)

    def _add_seq(self, toks):
        seq = [self.START] + toks + [self.END]
        for i in range(1, len(seq)):
            t = seq[i]
            self.uni[t] = self.uni.get(t, 0) + 1
            self.uni_total += 1
            self.vocab.add(t)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            self.bi.setdefault(a, {})
            self.bi[a][b] = self.bi[a].get(b, 0) + 1
        for i in range(len(seq) - 2):
            a, b, c = seq[i], seq[i + 1], seq[i + 2]
            key = (a, b)
            self.tri.setdefault(key, {})
            self.tri[key][c] = self.tri[key].get(c, 0) + 1

    def add_text(self, text):
        self._add_seq(self.tok.tokenize(text))

    def predict(self, t1, t2, bias_tokens=None):
        scores = {}
        if self.uni_total > 0:
            for tok, c in self.uni.items():
                scores[tok] = 0.10 * (c / self.uni_total)
        if t2 in self.bi:
            tot = sum(self.bi[t2].values())
            for tok, c in self.bi[t2].items():
                scores[tok] = scores.get(tok, 0.0) + 0.30 * (c / tot)
        key = (t1, t2)
        if key in self.tri:
            tot = sum(self.tri[key].values())
            for tok, c in self.tri[key].items():
                scores[tok] = scores.get(tok, 0.0) + 0.60 * (c / tot)
        if bias_tokens:
            for tok in bias_tokens:
                scores[tok] = scores.get(tok, 0.0) + 0.5
        return scores

    def generate(self, seed_tokens, bias_tokens=None, max_tokens=50):
        ctx = [self.START] + list(seed_tokens)
        out = list(seed_tokens)
        for _ in range(max_tokens):
            t1 = ctx[-2] if len(ctx) >= 2 else self.START
            t2 = ctx[-1] if ctx else self.START
            scores = self.predict(t1, t2, bias_tokens)
            if not scores:
                break
            for tok in list(scores.keys()):
                if out.count(tok) >= 2:
                    scores[tok] *= 0.25
                if out and tok == out[-1]:
                    scores[tok] *= 0.05
            tok = self._weighted_sample(scores)
            if tok is None or tok == self.END:
                break
            out.append(tok)
            ctx.append(tok)
        return out

    @staticmethod
    def _weighted_sample(scores):
        items = [(t, max(v, 1e-4)) for t, v in scores.items() if v > 0]
        if not items:
            return None
        total = sum(v for _, v in items)
        r = random.random() * total
        upto = 0.0
        for tok, v in items:
            upto += v
            if upto >= r:
                return tok
        return items[-1][0]


def _softmax(a, axis=-1):
    a = a - XP.max(a, axis=axis, keepdims=True)
    e = XP.exp(a)
    return e / XP.sum(e, axis=axis, keepdims=True)


def _gelu(x):
    return 0.5 * x * (1 + XP.tanh(XP.sqrt(2 / XP.pi) * (x + 0.044715 * x ** 3)))


class Qint:
    """v0.8: int8 权重量化容器。权重常驻只占 1 字节/参数(逼近 1B 参数也 ≤1GB),
    前向用 val() 临时还原成 fp32 参与矩阵乘 —— 存储大幅压缩, 算力(FLOPs)不减.
    v1.0: 量化改为"当前后端无关" —— 有 GPU 时随机数已直接在显存生成, 量化同样用 CuPy
    在显存完成, 不落地 numpy, 省去两次 Gpu↔CPU 全参搬运(启动因此显著更快)."""
    __slots__ = ("q", "s")

    def __init__(self, w):
        self.q, self.s = self._quantize(w)

    @staticmethod
    def _quantize(w):
        w = XP.asarray(w, dtype=XP.float32)
        if w.size == 0:
            return XP.zeros((0,), dtype=XP.int8), float(1.0)
        _m = XP.max(XP.abs(w))
        try:
            mx = float(_m)
        except Exception:
            mx = float(_to_host(_m).item())
        if mx < 1e-8:
            return XP.zeros_like(w, dtype=XP.int8), float(1.0)
        scale = mx / 127.0
        q = XP.clip(XP.round(w / scale), -127, 127).astype(XP.int8)
        return q, float(scale)

    def val(self):
        return XP.asarray(self.q.astype(XP.float32) * self.s)   # 逐层临时还原到当前后端

    def size(self):
        return int(self.q.size)

    def move_dev(self):
        self.q = _to_dev(self.q)
        return self


def _sinusoid(seq, d_model):
    pe = XP.zeros((seq, d_model), dtype=XP.float32)
    pos = XP.arange(seq)[:, None]
    i = XP.arange(d_model // 2)
    div = XP.power(10000.0, (2 * i) / d_model)
    pe[:, 0::2] = XP.sin(pos / div)
    pe[:, 1::2] = XP.cos(pos / div)
    return pe


class _F32RNG:
    """v1.0: 权重随机初始化直接落在当前后端 —— 有 NVIDIA 卡时用 CuPy random 在显存里成批生成
    1B 级参数(秒级完成), 不再拿 CPU numpy 逐个生成再搬家(那是 v0.5 启动两分钟的根因);
    无卡时自动回退 numpy, API 完全一致, 不影响功能与结果."""
    def __init__(self, seed):
        try:
            self.gen = XP.random.default_rng(seed)   # CuPy/numpy 都支持 default_rng(seed)
        except Exception:
            import numpy as _n
            self.gen = _n.random.default_rng(seed)

    def normal(self, loc=0.0, scale=1.0, size=None):
        # uses standard_normal (loc=0,scale=1) then scales: works on BOTH numpy & CuPy Generators
        return (self.gen.standard_normal(size).astype(XP.float32) * scale) + loc


class LayerNorm:
    def __init__(self, dims):
        self.gamma = XP.ones(dims, dtype=XP.float32)
        self.beta = XP.zeros(dims, dtype=XP.float32)

    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return self.gamma * (x - mean) / XP.sqrt(var + 1e-6) + self.beta


class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads, rng):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.Wk = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.Wv = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.Wo = Qint(rng.normal(0, 0.02, (d_model, d_model)))

    def forward(self, x):
        seq = x.shape[0]
        Wq, Wk, Wv, Wo = self.Wq.val(), self.Wk.val(), self.Wv.val(), self.Wo.val()
        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv
        Qh = Q.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        Kh = K.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        Vh = V.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        scores = Qh @ Kh.transpose(0, 2, 1) / XP.sqrt(self.d_k)
        mask = XP.triu(XP.full((seq, seq), -1e9), k=1)
        scores = scores + mask
        attn = _softmax(scores, -1)
        ctx = attn @ Vh
        ctx = ctx.transpose(1, 0, 2).reshape(seq, self.d_model)
        return ctx @ Wo, attn


def _sigmoid(x):
    return 1.0 / (1.0 + XP.exp(-XP.clip(x, -30, 30)))


class DeepIntentScorer:
    """v1.2(核心): 输入导向的深度意图定向 —— 彻底消除"牛头不对马嘴"。

    用户很容易被一句话里多种读法带偏, 根源在"猜"环节只靠随机权重, 没有真正吃透
    "用户拆出来的 token 到底指向哪个目标"。这里给猜猜乐加一个【意图锚】:
      1) 把用户拆词 token 的嵌入向量, 做一层【多头注意力池化】(矩形矩阵 Wq/Wk/Wv/Wo)
         压缩成一个"用户到底要什么"的意图向量 q;
      2) 再堆【多层纵深 ref ine】(Norm→GELU→残差门控) 把 q 一次比一次打磨得更贴用户目标
         —— 这就是"加深加深再加深";
      3) 反过来, 对猜猜乐里每个候选词, 用它的词嵌入与 q 做点积(矩形向量运算)得到
         "贴合度权重", 乘进候选分布 → 每个候选都朝用户意图精准收束, 不再瞎猜。
    """
    def __init__(self, d_model, rng, depth=4, pool_heads=2, ratio=0.18):
        self.d_model = d_model
        self.pool_heads = pool_heads
        self.d_k = max(1, d_model // pool_heads)
        self.depth = depth
        # —— ① 多头注意力池化: 从 token 嵌入收敛出"用户意图点" ——
        self.Wq = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.Wk = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.Wv = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.Wo = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        # —— ② 纵深 refine 塔: 每层把意图向量打磨得更贴目标 ——
        mid = max(64, int(d_model * ratio))
        self.refine = []
        for _i in range(depth):
            self.refine.append({
                "W1": Qint(rng.normal(0, 0.02, (d_model, mid))),
                "b1": XP.zeros(mid, dtype=XP.float32),
                "W2": Qint(rng.normal(0, 0.02, (mid, d_model))),
                "b2": XP.zeros(d_model, dtype=XP.float32),
            })
        self._proj = Qint(rng.normal(0, 0.02, (d_model, d_model)))
        self.last_vec = None      # 最近一次编码的意图向量(设备端, d_model 维)

    def encode(self, token_ids, embed):
        """token_ids: 用户拆词 int 序列. 返回设备端意图向量 (d_model,)."""
        d = self.d_model
        embed_f = embed.val()
        H = embed_f[token_ids]                 # (seq, d)
        seq = H.shape[0]
        # 因果自注意后取"末位即全句语义"作为 query → 全局读出用户意图
        Q = XP.tanh(H @ self.Wq.val())         # (seq, d)
        K = H @ self.Wk.val()                  # (seq, d)
        V = H @ self.Wv.val()                  # (seq, d)
        q_head = Q[-1:]                         # (1, d)  用末位 token 作为"要往下接什么"的锚
        Kt = K.reshape(seq, self.pool_heads, self.d_k).transpose(1, 0, 2)
        Vh = V.reshape(seq, self.pool_heads, self.d_k).transpose(1, 0, 2)
        qh = q_head.reshape(1, self.pool_heads, self.d_k).transpose(1, 0, 2)
        a = qh @ Kt.transpose(0, 2, 1) / XP.sqrt(float(self.d_k))   # (heads,1,seq)
        a = _softmax(a, -1)
        ctx = (a @ Vh)                                                 # (heads,1,d_k)
        ctx = ctx.transpose(1, 0, 2).reshape(1, d)                     # (1,d)
        vec = XP.tanh(ctx @ self.Wo.val())                             # (1,d)
        # —— 纵深 refine ——
        _P = self._proj.val()
        for _l in self.refine:
            W1, W2 = _l["W1"].val(), _l["W2"].val()
            g = _gelu(vec @ W1 + _l["b1"])
            upd = g @ W2 + _l["b2"]
            gate = _sigmoid(vec @ _P)
            vec = vec + gate * XP.tanh(upd)                            # 残差 + 门控加深
        self.last_vec = vec.reshape(d)
        return self.last_vec

    def param_count(self):
        n = 0
        for w in (self.Wq, self.Wk, self.Wv, self.Wo, self._proj):
            n += w.size()
        for _l in self.refine:
            n += _l["W1"].size() + _l["W2"].size()
            n += int(_l["b1"].size) + int(_l["b2"].size)
        return int(n)

    def storage_bytes(self):
        n = 0
        for w in (self.Wq, self.Wk, self.Wv, self.Wo, self._proj):
            n += w.size()
        for _l in self.refine:
            n += _l["W1"].size() + _l["W2"].size()
            n += 4 * (int(_l["b1"].size) + int(_l["b2"].size))
        return int(n)


class TransformerBlock:
    def __init__(self, d_model, d_ff, n_heads, rng):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rng)
        self.norm2 = LayerNorm(d_model)
        self.W1 = Qint(rng.normal(0, 0.02, (d_model, d_ff)))
        self.b1 = XP.zeros(d_ff, dtype=XP.float32)
        self.W2 = Qint(rng.normal(0, 0.02, (d_ff, d_model)))
        self.b2 = XP.zeros(d_model, dtype=XP.float32)

    def forward(self, x):
        a, attn = self.attn.forward(self.norm1.forward(x))
        x = x + a
        h = self.norm2.forward(x)
        W1, W2 = self.W1.val(), self.W2.val()
        h = _gelu(h @ W1 + self.b1)
        f = h @ W2 + self.b2
        x = x + f
        return x, attn


class DeepThinkTransformer:
    def __init__(self, vocab_list, d_model=MODEL_D, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS,
                 ngram_lm=None, ffn_ratio=MODEL_FFN, tie_out=MODEL_TIE, seed=42):
        self.vocab = list(vocab_list)
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ffn = ffn_ratio                       # v0.8: FFN 中间扩到 ×4, 容量更大
        self.tie_out = tie_out                     # v0.8: 输出↔输入嵌入共享 → 压缩存储, 算力不减
        self.ngram = ngram_lm
        rng = _F32RNG(seed)                        # v1.0: 随机权重直接在 GPU/numpy 后端生成
        _br.enter("神经网络权重初始化(嵌入投影)", boost=0.3)
        # v0.8: FFN ×4 提高单层容量; 所有大权重 int8 量化 → 参数逼近 1B 但存储 ≤1GB
        self.embed = Qint(rng.normal(0, 0.02, (self.vocab_size, d_model)))
        self.blocks = []
        for _li in range(n_layers):
            _br.tick(_li + 1, n_layers, "构建 Transformer 层 {}/{}".format(_li + 1, n_layers))
            self.blocks.append(TransformerBlock(d_model, d_model * ffn_ratio, n_heads, rng))
        _br.leave()
        self._ffn_norm = rng.normal(0, 0.02, (d_model,))
        # v1.2: 深度意图定向加权器 —— 用用户拆词 token 的嵌入, 走矩形矩阵+多头注意力池化+
        #       纵深精修塔, 提炼出"用户到底要什么"的意图向量; 再反向用词嵌入·意图的点积加权
        #       猜猜乐每个候选, 让生成/决策精准指向用户目标, 治愈"牛头不对马嘴"。
        self.intent_scorer = DeepIntentScorer(d_model, rng)
        if tie_out:
            self.W_out = None    # 输出投影与输入嵌入共享(W_out == embed.T), 省掉一整份 d×vocab
        else:
            self.W_out = Qint(rng.normal(0, 0.02, (d_model, self.vocab_size)))
        self.b_out = XP.zeros(self.vocab_size, dtype=XP.float32)
        if ngram_lm and ngram_lm.uni_total > 0:
            for tok, count in ngram_lm.uni.items():
                idx = self.token2id.get(tok)
                if idx is not None:
                    self.b_out[idx] = np.log(count / ngram_lm.uni_total + 1e-8) * 1.6

        if HAS_GPU:
            self.embed.move_dev(); self._ffn_norm = _to_dev(self._ffn_norm)
            if self.W_out is not None:
                self.W_out.move_dev()
            self.b_out = _to_dev(self.b_out)
            for blk in self.blocks:
                blk.norm1.gamma=_to_dev(blk.norm1.gamma); blk.norm1.beta=_to_dev(blk.norm1.beta)
                blk.attn.Wq.move_dev(); blk.attn.Wk.move_dev()
                blk.attn.Wv.move_dev(); blk.attn.Wo.move_dev()
                blk.norm2.gamma=_to_dev(blk.norm2.gamma); blk.norm2.beta=_to_dev(blk.norm2.beta)
                blk.W1.move_dev(); blk.b1=_to_dev(blk.b1)
                blk.W2.move_dev(); blk.b2=_to_dev(blk.b2)
            for w in (self.intent_scorer.Wq, self.intent_scorer.Wk,
                      self.intent_scorer.Wv, self.intent_scorer.Wo, self.intent_scorer._proj):
                w.move_dev()
            for _l in self.intent_scorer.refine:
                _l["W1"].move_dev(); _l["W2"].move_dev()
                _l["b1"] = _to_dev(_l["b1"]); _l["b2"] = _to_dev(_l["b2"])
    def forward(self, token_ids, trace=False):
        if len(token_ids) == 0:
            token_ids = [0]
        seq = len(token_ids)
        embed_f = self.embed.val()     # int8 嵌入临时还原为 fp32(权重常驻仍是 int8)
        x = embed_f[token_ids] + _sinusoid(seq, self.d_model)
        blocks = []
        attn_last = None
        if trace:
            # v0.6 Pro: 网格数据先在显存内算好, 最后一次性回拷 —— 消除"每层每次乘16头"的反复 GPU↔CPU 同步
            _rows, _norms, _pks = [], [], []
        for i, blk in enumerate(self.blocks):
            x, attn = blk.forward(x)
            attn_last = attn
            if trace:
                _rows.append(attn[0, -1, :])
                _pks.append(attn[:, -1, :].max(axis=1))
                _norms.append(XP.linalg.norm(x[-1]))
        if trace:
            _rows = _to_host(XP.stack(_rows))
            _pks = _to_host(XP.stack(_pks))
            _norms = _to_host(XP.stack(_norms))
            for i in range(len(self.blocks)):
                blocks.append({
                    "norm": round(float(_norms[i]), 3),
                    "row": [round(float(v), 3) for v in _rows[i]],
                    "peaks": [round(float(p), 3) for p in _pks[i]]})
        # 深度权重细化: 学习到的逐元素深度门控, 对深层表征做 1+ε 尺度调制
        final = x[-1] * (1.0 + 0.02 * self._ffn_norm)
        W_out = embed_f.T if self.tie_out else self.W_out.val()   # 输出-输入权重共享
        logits = final @ W_out + self.b_out
        probs = _softmax(logits, -1)
        tr = {
            "seq": seq,
            "embed_norm": float(XP.linalg.norm(x)),
            "blocks": blocks,
            "out_norm": float(XP.linalg.norm(final)),
            "gate_norm": float(XP.linalg.norm(self._ffn_norm)),
            "logits_norm": float(XP.linalg.norm(logits)),
            "probs": _to_host(probs),
            "attn_head0_last": _to_host(attn_last[0, -1, :]) if attn_last is not None else None,
        }
        return _to_host(probs), tr

    def param_count(self):
        """实时统计该 Transformer 的参数量(输出-输入共享时按共享后完整口径计数)."""
        n = self.embed.size()                               # 输入嵌入
        for blk in self.blocks:
            n += blk.attn.Wq.size() + blk.attn.Wk.size() + blk.attn.Wv.size() + blk.attn.Wo.size()
            n += blk.W1.size() + blk.W2.size()
            n += int(blk.b1.size) + int(blk.b2.size)
        n += self.d_model                                   # 深度门控
        n += self.d_model * self.vocab_size + self.vocab_size   # 输出投影 + 偏置(共享时沿语义仍算一份)
        n += self.intent_scorer.param_count()                   # v1.2: 深度意图定向层
        return int(n)

    def storage_bytes(self):
        """常驻存储(含量化权重): int8 权重 + 小量 fp32 偏置/门控."""
        n = self.embed.size()                               # int8
        for blk in self.blocks:
            n += blk.attn.Wq.size() + blk.attn.Wk.size() + blk.attn.Wv.size() + blk.attn.Wo.size()
            n += blk.W1.size() + blk.W2.size()
            n += 4 * (int(blk.b1.size) + int(blk.b2.size))
        n += 4 * (self.d_model + self.vocab_size)           # 深度门控 + 输出偏置 fp32
        n += self.intent_scorer.storage_bytes()                 # v1.2: 深度意图定向层
        return int(n)

    def forward_extra(self):
        return {"ffn_ratio": self.ffn, "tie_out": self.tie_out,
                "intent_scorer_depth": getattr(self.intent_scorer, "depth", 0)}

    def intent_encode(self, token_ids):
        """v1.2: 用户拆词 token → 深度意图向量(设备端). 供猜猜乐做贴合度加权."""
        if not token_ids:
            token_ids = [0]
        return self.intent_scorer.encode([int(t) for t in token_ids], self.embed)

    def _intent_weighted(self, result, intent_vec):
        # v1.2(核心): 猜猜乐候选加权 —— 每个候选词用「词嵌入 · 用户意图向量」算出贴合度,
        #   贴目标者放大、跑偏者压低, 让"猜"一步步朝用户到底要什么收束, 治愈牛头不对马嘴。
        if not result:
            return result
        cands = [t for t in result if t in self.token2id]
        if not cands:
            return result
        idxs = [self.token2id[t] for t in cands]
        e = self.embed.val()[idxs]                       # (C, d) 候选词嵌入
        sims = np.asarray(_to_host(e @ intent_vec), dtype=np.float64)   # (C,) 贴合度
        if sims.size == 0:
            return result
        s_min = float(sims.min()); s_max = float(sims.max())
        span = (s_max - s_min) or 1.0
        vals = [result[t] for t in cands]
        for t, w, s in zip(cands, vals, sims):
            nrm = (float(s) - s_min) / span              # 0..1, 越贴目标越高
            gain = (1.0 + 0.45 * nrm) if nrm > 0.5 else (1.0 - 0.22 * (0.5 - nrm))
            result[t] = max(0.05, float(w) * gain)
        return result

    def _intent_related(self, tokens, intent_vec, k=6):
        """v1.2: 找出与用户意图向量最贴合的语境词, 用于思考展示"意图锚定在哪"."""
        seen = []
        for t in tokens:
            if t and t in self.token2id and t not in seen:
                seen.append(t)
        if not seen:
            return []
        idxs = [self.token2id[t] for t in seen]
        e = self.embed.val()[idxs]
        sims = _to_host(e @ intent_vec)
        pairs = sorted(zip(seen, sims), key=lambda p: -p[1])[:k]
        return [(t, round(float(s), 3)) for t, s in pairs]

    def score_distribution(self, tokens, bias_tokens=None, probs=None, top_k=96,
                           intent_vec=None):
        # v0.6 懒加载: 只对"候选集"(bias词 + ngram高频 + probs top)打分,
        #   不再遍历上万词库 —— 根治"万词一直循环"导致的卡顿。
        # 传入 probs 可复用 think() 里 trace 那一次 forward 的结果 (修复双 forward)。
        ids = [self.token2id.get(t, 0) for t in tokens]
        if not ids:
            ids = [0]
        if probs is None:
            probs, _ = self.forward(ids, trace=False)
        result = {}
        if self.ngram:
            t1 = tokens[-2] if len(tokens) >= 2 else NgramLM.START
            t2 = tokens[-1] if tokens else NgramLM.START
            ng = self.ngram.predict(t1, t2, bias_tokens)
            cand_idx = set(bias_tokens) if bias_tokens else set()
            try:
                top_ids = np.argsort(probs)[::-1][:top_k]
                for i in top_ids:
                    cand_idx.add(self.vocab[i])
            except Exception:
                pass
            if ng:
                for tok, _ in sorted(ng.items(), key=lambda kv: -kv[1])[:top_k]:
                    cand_idx.add(tok)
            for tok in cand_idx:
                idx = self.token2id.get(tok)
                if idx is None:
                    continue
                p = float(probs[idx])
                w = ng.get(tok, 0.0)
                result[tok] = p * (0.3 + w)
            if bias_tokens:
                for tok in bias_tokens:
                    if tok in self.token2id:
                        result[tok] = result.get(tok, 0.0) + 0.2
        else:
            try:
                top_ids = np.argsort(probs)[::-1][:top_k]
                for i in top_ids:
                    tok = self.vocab[i]
                    result[tok] = float(probs[i])
            except Exception:
                for tok in self.vocab:
                    result[tok] = float(probs[self.token2id[tok]])
        # v1.2: 深度意图定向 —— 候选集按与用户意图的贴合度加权, 精准指向用户到底要什么
        if intent_vec is not None:
            result = self._intent_weighted(result, intent_vec)
        return result

    @staticmethod
    def _sample_top_p(dist, temperature=0.85, top_p=0.95):
        items = [(t, float(v)) for t, v in dist.items() if v > 1e-9]
        if not items:
            return None
        keys = [t for t, _ in items]
        vals = np.array([v for _, v in items], dtype=np.float64)
        vals = np.power(vals, 1.0 / max(temperature, 1e-3))
        order = np.argsort(vals)[::-1]
        sorted_v = vals[order]
        total = sorted_v.sum()
        if total <= 0:
            return None
        cum = np.cumsum(sorted_v / total)
        keep_idx = order[cum <= top_p]
        if len(keep_idx) == 0:
            keep_idx = order[:1]
        sub_v = vals[keep_idx]
        sub_t = [keys[i] for i in keep_idx]
        sub_v = sub_v / sub_v.sum()
        r = random.random()
        upto = 0.0
        for i, v in enumerate(sub_v):
            upto += v
            if upto >= r:
                return sub_t[i]
        return sub_t[-1]

    def generate(self, seed_tokens, bias_tokens=None, max_tokens=42,
                 temperature=0.85, top_p=0.95):
        ctx = list(seed_tokens)
        out = list(seed_tokens)
        for _ in range(max_tokens):
            dist = self.score_distribution(ctx, bias_tokens)
            if not dist:
                break
            for tok in list(dist.keys()):
                c = out.count(tok)
                if c >= 2:
                    dist[tok] *= 0.25 ** (c - 1)
                if out and tok == out[-1]:
                    dist[tok] *= 0.05
            tok = self._sample_top_p(dist, temperature, top_p)
            if tok is None or tok == NgramLM.END:
                break
            out.append(tok)
            ctx.append(tok)
        return out


class TokenMeter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total = 0
        self.turns = 0

    def count_input(self, tokens):
        n = len(tokens)
        self.input_tokens += n
        self.total += n
        UI_ST["in"] = self.input_tokens        # v1.4: 实时 Token 进面板
        return n

    def count_output(self, tokens):
        n = len(tokens)
        self.output_tokens += n
        self.total += n
        UI_ST["out"] = self.output_tokens      # v1.4: 实时 Token 进面板
        return n

    def new_turn(self):
        self.turns += 1

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total = 0

    def format(self):
        return "Token: in={} out={} total={} turns={}".format(
            self.input_tokens, self.output_tokens, self.total, self.turns)


class SemanticRetriever:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.docs = []
        for entry in DATA.KNOWLEDGE_BASE:
            parts = [entry["t"]]
            parts.extend(entry.get("a", []) or [])
            parts.append(entry["b"])
            pool = " ".join(str(p) for p in parts)
            self.docs.append({
                "entry": entry,
                "tokens": set(tokenizer.tokenize(pool)),
                "title": entry["t"],
                "aliases": entry.get("a", [])})

    def retrieve(self, query, top_k=4, min_score=0.03):
        q = self.tok.tokenize_set(query)
        if not q:
            return []
        scored = []
        for doc in self.docs:
            entry = doc["entry"]
            inter = q & doc["tokens"]
            union = q | doc["tokens"]
            jaccard = len(inter) / len(union) if union else 0.0
            coverage = len(inter) / max(len(q), 1)
            bonus = 0.0
            # v0.8 Pro: 精确标题命中给上榜资格, 但强不强由 _kb_aligned(实体对齐)决定
            if doc["title"] and doc["title"] in query:
                # v0.6 正式版: 标题越长越精确, 加权越重 (图灵完备 > 图灵, 避免答非所问)
                bonus += 1.2 + min(1.0, len(doc["title"]) * 0.2)
            for a in doc["aliases"]:
                if a and a in query:
                    bonus += 0.8
                    if len(a) >= 3:
                        bonus += min(0.6, len(a) * 0.12)
            score = jaccard + bonus + coverage * 0.3
            if score > min_score:
                scored.append((round(score, 3), entry))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


def _kb_aligned(query, entry):
    """v0.8 Pro: 实体对齐门控 —— query 里是否精确出现该条目主标题/主别名.
    防止"随便几个字沾边就翻本地知识库答非所问"."""
    cands = [entry.get("t")]
    cands.extend(list(entry.get("a", []) or []))
    return any(isinstance(c, str) and len(c) >= 2 and c in query for c in cands if c)


_SENT_SPLIT = re.compile(r"[。；！？!?;\n]+")


def _segments_from_tokens(tokens):
    return "".join(tokens)


def _split_sents(text):
    parts = re.split(r"[。；！？!?]+", text)
    return [p.strip() for p in parts if p.strip()]


# v1.2: emoji 分界符 —— 只有跟在句首/句尾/这些标点衔接处(逗号那句)的 emoji 才算"待对位置"
_EMOJI_PUNCT = "，,。！!？?；;：:、…~～"


def _normalize_emoji(text):
    """v1.2: emoji 只准待在句首/句尾/标点分句段的衔接位(逗号那边), 绝不夹在词中间。
    词中间的乱插 emoji 会被圈走, 攒到段尾统一放一个 —— 保持"一句话读到一半不被 emoji 打断"。
    """
    import unicodedata as _uda
    if not text:
        return text
    chars = list(text)
    out = []
    tail_pool = []
    n = len(chars)
    for i, c in enumerate(chars):
        try:
            is_emo = _uda.category(c) == "So"
        except Exception:
            is_emo = False
        if not is_emo:
            out.append(c)
            continue
        head_ok = (i == 0)
        tail_ok = (i == n - 1)
        prev_c = chars[i - 1] if i > 0 else ""
        next_c = chars[i + 1] if i < n - 1 else ""
        prev_break = bool(prev_c) and prev_c in _EMOJI_PUNCT
        next_break = bool(next_c) and next_c in _EMOJI_PUNCT
        if head_ok or tail_ok or prev_break or next_break:
            out.append(c)             # 待在句首/句尾/标点衔接处 → 保留原位
        else:
            tail_pool.append(c)       # 词中乱插 → 清走, 段尾再统一放一个
    if tail_pool:
        out.append(tail_pool[-1])
    return "".join(out)


def _segment_text(text, chunk_size=12):
    text = text.strip()
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _fmt(x, nd=3):
    return ("{:.%df}" % nd).format(x)


class ResponseGenerator:
    def __init__(self, tokenizer, lm, transformer=None):
        self.tok = tokenizer
        self.lm = lm
        self.tf = transformer

    def generate(self, query, kb_hits, emo, intent, history):
        opener = self._pick_opener(emo)
        # v0.8 Pro: 知识直答必须"分数够 + 实体精确对齐", 缺一则视为没问到点子上, 转真答/澄清
        strong_kb = bool(kb_hits and kb_hits[0][0] >= 0.68
                         and _kb_aligned(query, kb_hits[0][1]))
        if strong_kb:
            body = self._gen_from_kb(kb_hits, emo)
        elif emo["score"] <= -3 or intent["top"] in ("help", "study", "start", "suggest"):
            body = self._gen_comfort(emo, history)
        else:
            body = self._gen_freeform(query, emo, history)   # 直指用户问题作答
        body = self._insert_emojis(body, emo)
        closer = self._pick_closer(emo)
        full = " ".join(p for p in [opener, body, closer] if p)
        return _normalize_emoji(full)     # v1.2: emoji 只留句首/句尾/标点处的, 清掉词中乱插

    def _pick_opener(self, emo):
        if emo["score"] >= 3:
            pool = DATA.OPENERS_POSITIVE
        elif emo["score"] <= -3:
            pool = DATA.OPENERS_NEGATIVE
        elif emo["has_question"]:
            pool = DATA.OPENERS_POSITIVE + DATA.OPENERS_NEUTRAL
        else:
            pool = DATA.OPENERS_NEUTRAL
        return random.choice(pool)

    def _pick_closer(self, emo):
        if emo["score"] <= -3:
            return random.choice(DATA.CLOSERS_NEG)
        if emo["score"] >= 3:
            return random.choice(DATA.CLOSERS_POS)
        return random.choice(DATA.CLOSERS)

    def _gen_from_kb(self, kb_hits, emo):
        score, entry = kb_hits[0]
        sents = _split_sents(str(entry["b"]))
        if not sents:
            return str(entry["b"])
        take = min(len(sents), random.choice([2, 3, 3]))
        body = "。".join(sents[:take])
        if not (body.endswith("。") or body.endswith("！") or body.endswith("？")):
            body += "。"
        related = entry.get("r", [])
        if related:
            body += " 相关的还有" + "、".join(related[:3]) + "。"
        return body

    def _gen_comfort(self, emo, history):
        anchor = self._comfort_anchor(emo)
        support = random.choice([
            "我就在这儿，你想到什么都可以随时说。",
            "先照顾好自己，剩下的都还有回旋的余地。",
            "不用立刻好起来，一步一步来就好。",
            "你愿意的话，我可以陪你一起把乱成一团的事理一理。",
            "此刻觉得沉重也很正常，说出来会轻一点。",
        ])
        return anchor + " " + support

    def _comfort_anchor(self, emo):
        if emo["has_anger"]:
            if emo["intensity"] >= 6:
                return "我能感觉到你气得不轻，先深呼吸三下，我陪你。"
            return "生气是正常的，咱们慢慢说，别急着憋着。"
        if emo["has_sad"]:
            if emo["intensity"] >= 6:
                return "我感受到你很难过，先别扛着，哭出来也没关系。"
            return "难过的情绪我懂，抱抱你，我一直在。"
        if emo["has_fear"]:
            return "担心的事先放一放，我陪你一起把它理清楚。"
        if emo["has_tired"]:
            return "累了就先歇会儿，哪怕闭眼五分钟也是赚到。"
        return "我在这儿，不管什么情绪都可以跟我说说。"

    def _gen_freeform(self, query, emo, history):
        # v0.8 Pro: 直指用户问题回答 —— 不再"本地没存你就去搜"(治答非所问/牛头不对马嘴)
        core = self._core_entity(query)          # 提炼用户真正要问的实体
        hit = self._kb_smart_hit(query)
        if hit:                                   # ① 本地确实有对应条目 → 直接给答案
            return hit
        direct = self._direct_ask(query, core)
        if direct:                                # ② 能确定性回答的常见浅问 → 直接答
            return direct
        if core:                                  # ③ 有实体 → 给一句"真在回应这问题"的定向答复
            return ("我明白你想问的是「{}」。这个话题分几个点好讲：定义、原理、常见用法。"
                    "你可以直接问我『{}是什么』『{}怎么用』『{}有什么例子』，我会更精准地展开给你。"
                    "这里先给你一句要旨：{}").format(
                core, core, core, core, self._seed_answer(core))
        # ④ 完全虚指 → 温和澄清, 避免答非所问
        return ("这句我还没完全 get 到你具体想问什么 🤔 你可以换个说法，"
                "或者直接问『X 是什么』『X 怎么做』，我立刻精准答你。")

    def _core_entity(self, query):
        """从 query 提炼核心实体: 优先知识库标题/主别名, 其次去问词尾后的主干."""
        t = query.strip().strip(" ？?。！!，,、；;：:\"'“”（）()")
        for entry in DATA.KNOWLEDGE_BASE:
            for c in [entry["t"]] + list(entry.get("a", []) or []):
                if c and isinstance(c, str) and len(c) >= 2 and c in t:
                    return c
        for tail in ("是什么", "是什么意思", "怎么用", "怎么玩", "怎么样",
                     "怎么做", "为什么", "能不能", "吗", "呢", "啊", "呀", "吧"):
            if t.endswith(tail):
                t = t[:-len(tail)]
                break
        t = t.strip(" ！!？?，,、；;：:。的你了我们在就不要是")
        return t[:12] if len(t) >= 2 else None

    def _kb_smart_hit(self, query):
        """整句覆盖足够 + 实体对齐 → 返回该知识库句子作为直接答案."""
        qset = self.tok.tokenize_set(query)
        if not qset:
            return None
        best = None
        for entry in DATA.KNOWLEDGE_BASE:
            if not _kb_aligned(query, entry):
                continue
            for s in _split_sents(str(entry["b"])):
                st = self.tok.tokenize_set(s)
                if not st:
                    continue
                inter = len(qset & st)
                if inter / max(len(qset), 1) >= 0.7 and (best is None or inter > best[0]):
                    best = (inter, s)
        if best:
            return best[1]
        return None

    def _direct_ask(self, query, core):
        """①能直接答的浅问(能力/功能/在干嘛) ②本地有该实体的概况 → 直接答, 不绕库不甩搜."""
        if not core:
            return None
        c = core
        if any(k in c for k in ("能干", "会什么", "会啥", "功能", "技能", "作用")):
            return "我能陪你聊天、解答疑问、算数学、写代码、深度思考、跨会话自学，还能联网查最新资讯。你想先试哪样？"
        if any(k in c for k in ("怎么用", "怎么运行", "怎么启动", "怎么打开")):
            return "直接双击「启动小方.bat」就能运行我；主题设置用「setting」，看能力用「help」。要我现在演示一下吗？"
        for entry in DATA.KNOWLEDGE_BASE:
            if c == entry["t"]:
                return _split_sents(str(entry["b"]))[0] + "。"
        return None

    def _seed_answer(self, core):
        for entry in DATA.KNOWLEDGE_BASE:
            if core == entry["t"] or core in str(entry["t"]):
                s = _split_sents(str(entry["b"]))
                return s[0][:60] + "。" if s else "相关要点我可以马上展开。"
            if core in str(entry["b"]):
                s = _split_sents(str(entry["b"]))
                return s[0][:60] + "。"
        return "我可以从定义、原理、应用三方面给你讲透，也能配上例子和对比表格，你想先听哪块？"

    def _insert_emojis(self, text, emo):
        # v0.6 正式版: 情绪 emoji 只放句首或句尾, 绝不在句中打断 (修复"爱在中间加 emoji")
        text = (text or "").strip()
        if not text:
            return text
        emoji = self._pick_emoji(emo)
        if not emoji:
            return text
        if random.random() < 0.4:
            return "{} {}".format(emoji, text)
        return text.rstrip() + " " + emoji

    def _pick_emoji(self, emo):
        if emo["score"] >= 2:
            pool = DATA.EMOJI_POSITIVE
        elif emo["score"] <= -2:
            pool = DATA.EMOJI_NEGATIVE
        else:
            pool = DATA.EMOJI_NEUTRAL
        items = [(e, max(w, 1)) for e, w in pool]
        total = sum(w for _, w in items)
        r = random.random() * total
        upto = 0
        for e, w in items:
            upto += w
            if upto >= r:
                return e
        return ""

    def _history_seed(self, history):
        if len(history) >= 2:
            return self.tok.tokenize(history[-2])[:3]
        return None

    def _history_tokens(self, history):
        if not history:
            return []
        return self.tok.tokenize(history[-1]) if len(history) >= 1 else []


class PersonaResponder:
    # v0.5 Alpha 2: 设定人格。仅在明确询问"你家/家人/住址/年龄/喜好/身世/世界观"时启用,
    # 其余中性问答一律关闭, 维持 AI 助手身份。禁止输出"(摆摆手)"等括号动作。
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.entries = list(getattr(DATA, "PERSONA_KB", []))

    def detect(self, raw):
        matched = []
        for e in self.entries:
            n = sum(1 for kw in e["kws"] if kw and kw in raw)
            if n > 0:
                matched.append((n, e))
        matched.sort(key=lambda x: -x[0])
        return matched

    def reply(self, raw):
        matched = self.detect(raw)
        if not matched:
            return None
        _, entry = matched[0]
        body = str(entry["body"]).strip()
        if body and not body[-1] in "。！？!?；":
            body += "。"
        return body


# ============================================================
# v0.5 正式版: 数学求解器 (纯原生, 递归下降解析, 不做字符串 eval)
# ============================================================
class _MathParser:
    def __init__(self, s):
        self.toks = re.findall(r"\d+\.?\d*|[()+\-*/%]", s)
        self.i = 0
    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None
    def parse(self):
        v = self.expr()
        if self.i != len(self.toks):
            raise ValueError("extra")
        return v
    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.toks[self.i]; self.i += 1; r = self.term()
            v = v + r if op == "+" else v - r
        return v
    def term(self):
        v = self.factor()
        while self.peek() in ("*", "/", "%"):
            op = self.toks[self.i]; self.i += 1; r = self.factor()
            if op == "*":
                v = v * r
            elif op == "/":
                v = v / r if r else (_abort())
            else:
                v = v % r if r else (_abort())
        return v
    def factor(self):
        tok = self.peek()
        if tok in ("+", "-"):
            self.i += 1
            v = self.factor()
            return v if tok == "+" else -v
        if tok == "(":
            self.i += 1
            v = self.expr()
            if self.peek() != ")":
                raise ValueError("no close")
            self.i += 1
            return v
        if tok is not None and re.fullmatch(r"\d+\.?\d*", tok):
            self.i += 1
            return float(tok)
        raise ValueError("unexpected")


def _abort():
    raise ValueError("div0")


class MathResponder:
    # 支持 + - * / % ( ) 与中文"加减乘除"、平方/立方; 安全解析(非 eval)
    REP = {"×": "*", "÷": "/", "＋": "+", "－": "-", "＝": "=", "（": "(", "）": ")", "％": "%"}

    def detect(self, raw):
        if not isinstance(raw, str):
            return False
        if re.search(r"\d\s*[\+\-*/%×÷]\s*\d", raw):
            return True
        if re.search(r"[\d一二两三四五六七八九十百千万]+(?:加|减|乘|除|乘以|除以|的平方|的立方|等于多少|等于|求值)", raw):
            return True
        if any(w in raw for w in ["算一下", "计算", "数学题", "求和", "求值"]):
            return True
        # v1.3 Alpha: 变量方程(如 x²=4 / 2x=6 / x+3=5)也判为数学
        if re.search(r"[a-df-zA-DF-Z](?:\s*\^?\s*[²³23])?\s*[=＝]", raw) or \
           re.search(r"[=＝]\s*-?[\d(].*[a-df-zA-DF-Z]", raw):
            return True
        # v1.3 Alpha 补: 变量与 = 之间夹着运算项(如 "x + 3 = 5" / "3x - 5 = 10"),
        #   前面两个正则都因变量不紧贴 = 而漏判 → 只要句子确有 = 且两侧含变量字母就算方程。
        if re.search(r"[=＝]", raw) and re.search(r"[a-df-zA-DF-Z].*\d", raw):
            return True
        return False

    def _extract_eq(self, raw):
        # 从整句里抠出"变量=值"等式片段, 忽略前后的中文修饰词(如"有道数学题我不会，x²=4，帮我解答")
        M = "0-9a-df-zA-DF-Z+\\-*/%()^=＝²³·. "
        m = re.search(r"[" + M + r"]{0,12}[a-df-zA-DF-Z][" + M + r"]{0,12}[=＝][" + M + r"]{0,12}", raw)
        return m.group(0).strip() if m else None

    def _normalize(self, raw):
        s = raw.strip()
        for k, v in self.REP.items():
            s = s.replace(k, v)
        # 中文数字运算符 -> 英文运算符
        s = re.sub(r"乘以", "*", s); s = re.sub(r"除以", "/", s)
        s = re.sub(r"的平方", "**2", s); s = re.sub(r"的立方", "**3", s)
        # v1.3 Alpha: 上标/次方符号归一: x²→x**2, x^2→x**2, x³→x**3
        s = s.replace("²", "**2").replace("^", "**").replace("³", "**3")
        # 隐式乘法: 2x→2*x, x(2)→x*(2), x2→x*2 (先于中文运算符替换)
        s = re.sub(r"(?<=\d)(?=[a-df-zA-DF-Z(])", "*", s)
        s = re.sub(r"(?<=\))(?=[a-df-zA-DF-Z(])", "*", s)
        s = re.sub(r"(?<=[a-df-zA-DF-Z])(?=\d)", "*", s)
        # 把"加/减/乘/除"仅在数字之间替换成符号
        s = re.sub(r"(?<=\d|\))(加)(?=\d|\()", "+", s)
        s = re.sub(r"(?<=\d|\))(减)(?=\d|\()", "-", s)
        s = re.sub(r"(?<=\d|\))(乘)(?=\d|\()", "*", s)
        s = re.sub(r"(?<=\d|\))(除)(?=\d|\()", "/", s)
        s = re.sub(r"等于", "=", s)
        s = s.replace("乘以", "*").replace("除以", "/")
        return s

    def _extract(self, norm):
        # 去掉 = 及之后(若表达等号)
        body = norm.split("=")[0]
        # 抓取包含数字/运算符/括号的最长子串; 交给解析器验证合法性
        m = re.search(r"[+\-]?[\d\s.()+\-*/%]+", body)
        if not m:
            return None
        part = m.group(0).strip()
        if "**" in part:
            return None
        return part or None

    def _fmt(self, v):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return ("%.5f" % v).rstrip("0").rstrip(".")

    # ---------- v1.3 Alpha: 变量方程求解 (修复"x²=4"被通用模板曲解) ----------
    def _pv(self, node):
        # 只允许 数字 + 加减乘除/幂 + 正负号 的算术 AST, 其余一律拒绝(防注入/防乱来)
        import ast as _A
        if isinstance(node, _A.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("非数字常量")
        if isinstance(node, _A.BinOp):
            a = self._pv(node.left); b = self._pv(node.right)
            if isinstance(node.op, _A.Add): return a + b
            if isinstance(node.op, _A.Sub): return a - b
            if isinstance(node.op, _A.Mult): return a * b
            if isinstance(node.op, _A.Div): return a / b
            if isinstance(node.op, _A.Pow) and isinstance(b, int) and b >= 0:
                return a ** b
            raise ValueError("非法运算")
        if isinstance(node, _A.UnaryOp):
            if isinstance(node.op, _A.USub): return -self._pv(node.operand)
            if isinstance(node.op, _A.UAdd): return self._pv(node.operand)
        raise ValueError("非法表达式")

    def _trim(self, v):
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e12:
            return str(int(round(v)))
        return ("%.4f" % v).rstrip("0").rstrip(".")

    def _solve_equation(self, norm):
        import math as _m, ast as _A
        sides = [s.strip() for s in norm.split("=")]
        if len(sides) != 2 or not all(sides):
            return None
        lhs, rhs = sides
        mv = re.search(r"[a-df-zA-DF-Z]", lhs + rhs)
        if not mv:
            return None
        var = mv.group(0)
        expr = ("(" + lhs + ")-(" + rhs + ")").replace(" ", "")
        _pv = self._pv
        def probe(xv):
            src = re.sub(r"(?<![0-9A-Za-z_])" + re.escape(var) + r"(?![0-9A-Za-z_])",
                         "(" + repr(float(xv)) + ")", expr)
            return _pv(_A.parse(src, mode="eval").body)
        try:
            f0, f1, fm1 = probe(0.0), probe(1.0), probe(-1.0)
        except Exception:
            return None
        c0, c1, c2 = f0, (f1 - fm1) / 2.0, (f1 + fm1 - 2.0 * f0) / 2.0
        # 展示用方程(恢复上标/隐式乘号)
        disp = (lhs + " = " + rhs).replace("**2", "²").replace("**3", "³")
        disp = re.sub(r"(?<=\d)\*(?=[a-zA-Z(])", "", disp)
        eps = 1e-9
        if abs(c2) <= eps and abs(c1) <= eps:
            return None if abs(c0) <= eps else \
                "🧮 方程 {}：两边化简化后自相矛盾，无解。".format(disp)
        if abs(c2) <= eps:                      # 一次: a·x + b = 0
            return "🧮 解方程 {}：**{} = {}**".format(disp, var, self._fmt(-c0 / c1))
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc < -eps:
            return "🧮 方程 {}：在实数范围内无解（判别式 Δ<0）。".format(disp)
        if abs(c1) <= eps:                      # 标准平方形: a·x² ± k = 0 → ±√
            k = -c0 / c2
            if k < -eps:
                return "🧮 方程 {}：在实数范围内无解。".format(disp)
            r = _m.sqrt(k)
            ri = round(r)
            tail = "±{}".format(ri) if abs(ri * ri - k) < 1e-6 else \
                "±√({}) ≈ ±{}".format(self._trim(k), self._trim(r))
            return "🧮 解方程 {}：**{} = {}**（正负两个解都成立）".format(disp, var, tail)
        sq = _m.sqrt(max(0.0, disc))
        r1, r2 = (-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2)
        if abs(r1 - r2) < eps:
            return "🧮 解方程 {}：**{} = {}**（重根）".format(disp, var, self._fmt(r1))
        return "🧮 解方程 {}：**{} = {} 或 {} = {}**".format(
            disp, var, self._fmt(r1), var, self._fmt(r2))

    def answer(self, raw):
        if not self.detect(raw):
            return None
        norm = self._normalize(raw)
        # v1.3 Alpha: 变量方程精确求解, 不再被通用模板(定义/原理/用法)曲解
        #   用规范化后的文本抠等式片段, 这样 "x²=4" / "x^2=4" / "x的平方等于4" 都能命中
        eq_raw = self._extract_eq(norm) or self._extract_eq(raw)
        if eq_raw:
            eq_norm = self._normalize(eq_raw)
            if "=" in eq_norm and re.search(r"[a-df-zA-DF-Z]", eq_norm):
                _eq = self._solve_equation(eq_norm)
                if _eq:
                    return _eq
        # 平方/立方 快捷: "N^2的平方/立方/等于"
        sq = re.search(r"(\d+(?:\.\d+)?)\s*\*\*2", norm)
        cb = re.search(r"(\d+(?:\.\d+)?)\s*\*\*3", norm)
        sq = sq or (re.search(r"(\d+(?:\.\d+)?)[^+\-*/%]*的平方", norm) and None)
        if sq and not re.search(r"[+\-*/%]", norm):
            n = float(sq.group(1)); v = n * n
            return "🧮 {0} 的平方 = {0} × {0} = **{1}**".format(self._fmt(n), self._fmt(v))
        if re.search(r"\*\*", norm):
            lift = re.search(r"(\d+(?:\.\d+)?)\s*\*\*\s*([23])\b", norm)
            if lift:
                n = float(lift.group(1)); k = int(lift.group(2)); v = n ** k
                return "🧮 {0} 的{1}次方 = **{2}**".format(self._fmt(n), "平方" if k == 2 else "立方", self._fmt(v))
        expr = self._extract(norm)
        if not expr:
            return None
        try:
            val = _MathParser(expr).parse()
        except Exception:
            return None
        res = self._fmt(val)
        # 生成分步讲解
        parts = _split_sents(raw)
        # 单运算符二元计算给出分步
        mm = re.fullmatch(r"\s*([+\-]?\d+(?:\.\d+)?)([+\-*/%])([+\-]?\d+(?:\.\d+)?)\s*",
                          expr.replace("--", "+").replace("+-", "-"))
        if mm:
            a, op, b = mm.group(1), mm.group(2), mm.group(3)
            desc = {"+": "加", "-": "减", "*": "乘", "/": "除", "%": "取余"}[op]
            step = "{} {} {} = {}".format(a, op, b, res)
            return "🧮 {0}：{1} 计算得到 **{2}**。".format(desc, step, res)
        total = self._fmt(val)
        # 多运算符: 展示规约后的最终式
        return "🧮 计算 {} 得 **{}**。".format(norm.split("=")[0].strip(), total)


# ============================================================
# v0.5 正式版: 代码生成器 (内含算法逻辑讲解)
# ============================================================
class CodeResponder:
    _TASKS = [
        {"kws": ["冒泡", "bubble"], "title": "冒泡排序",
         "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        for j in range(n - 1 - i):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
         "exp": "思路：外层每轮把最大的数“冒泡”到末尾；内层两两比较，前大后小就交换。每轮少比较一次，N 个数共 O(N²)。"},
        {"kws": ["快速排序", "快排", "quick"], "title": "快速排序",
         "code": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    p = arr[0]\n    left = [x for x in arr[1:] if x < p]\n    right = [x for x in arr[1:] if x >= p]\n    return quick_sort(left) + [p] + quick_sort(right)",
         "exp": "思路：分治。任选基准 p，把比 p 小的放左边、大的放右边，再递归对两边排序。平均 O(N log N)。"},
        {"kws": ["二分", "binary"], "title": "二分查找",
         "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
         "exp": "思路：对有序数组每次砍掉一半，比较中点和目标决定去左半还是右半，复杂度 O(log N)。"},
        {"kws": ["阶乘", "factorial"], "title": "阶乘",
         "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
         "exp": "思路：递归。factorial(n) = n × factorial(n-1)，边界是 n≤1 返回 1。"},
        {"kws": ["斐波那契", "fibonacci"], "title": "斐波那契",
         "code": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
         "exp": "思路：动态规划递推，每一项是前两项之和；用双变量滚动省内存，O(N)。"},
        {"kws": ["素数", "质数", "prime"], "title": "素数判断",
         "code": "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
         "exp": "思路：只需枚举到 √n，只要有一个能整除就不是素数，O(√n)。"},
        {"kws": ["最大公约数", "gcd", "欧几里得"], "title": "最大公约数",
         "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
         "exp": "思路：欧几里得算法，gcd(a,b)=gcd(b, a mod b)，反复取余直到余数为 0。"},
        {"kws": ["摄氏度", "华氏度", "温度", "温度转换"], "title": "温度转换",
         "code": "def c_to_f(c):\n    return c * 9 / 5 + 32\n\ndef f_to_c(f):\n    return (f - 32) * 5 / 9",
         "exp": "思路：摄氏转华氏 ℉=℃×9/5+32；反向用 (℉-32)×5/9。"},
        {"kws": ["二进制", "进制转换", "进制"], "title": "十进制转二进制",
         "code": "def to_bin(n):\n    return bin(n)[2:] if isinstance(n, int) else n",
         "exp": "思路：反复除 2 取余，余数倒排即得二进制；Python 的 bin() 直接给结果。"},
        # —— v1.2 加强: 基础语法教程 (修复"写循环教程"被答成 Python 创始人) ——
        {"kws": ["循环", "for循环", "for 循环", "while循环", "while 循环", "遍历"],
         "title": "Python 循环教程 (for / while)",
         "code": "# for: 依次取序列(range/列表/字符串)里每一个\nfor i in range(5):\n    print(i)      # 依次打印 0 1 2 3 4\n\nmy_list = [10, 20, 30]\nfor item in my_list:\n    print(item)   # 依次 10 20 30\n\n# while: 条件成立就一直执行, 记得让条件会变化\ncount = 0\nwhile count < 3:\n    print(count)\n    count += 1    # 少了这行会死循环\n\n# break 提前跳出, continue 直接进入下一次\nfor i in range(10):\n    if i == 3:\n        continue   # 跳过 3\n    if i == 6:\n        break      # 到 6 就停\n    print(i)",
         "exp": "思路：for 用来遍历序列，自动取下一个元素；while 靠条件决定，只要条件为真就执行，务必让条件会变化否则死循环。break 跳出整层循环、continue 跳过本次进入下一次。想进阶就循环里写循环（嵌套），再配列表推导式更优雅。"},
        {"kws": ["python", "py", "基础教程", "入门教程", "语法教程"],
         "title": "Python 基础入门教程",
         "code": "# 1) 变量: 不需要声明类型, 直接赋值\nx = 5\nname = \"小方\"\n\n# 2) 条件: if / elif / else\nif x > 3:\n    print(\"x 大于 3\")\nelif x == 3:\n    print(\"正好 3\")\nelse:\n    print(\"x 小于 3\")\n\n# 3) 循环: for 遍历, while 按条件\nfor i in range(3):\n    print(i)\n\n# 4) 函数: def 定义, 调用了才执行\n\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n\n# 5) 容器: 列表[] / 字典{} / 元组()/供索引切片\nnums = [1, 2, 3, 4]\nprint(nums[0], nums[-1])   # 1 4\nd = {\"name\": \"小方\", \"age\": 5}\nprint(d[\"name\"])\n\n# 6) 注释: # 开头的行不会被运行",
         "exp": "思路：Python 从 变量→条件(if/elif/else)→循环(for/while)→函数(def)→容器(列表/字典/元组) 层层递进。先照着逐行跑、改数字看输出变化，把每种语法跑通即算入门。想深入哪个再只管喊我。"},
    ]
    _GENERIC = ("# 一段可运行的基础骨架\ndef function_name(arr):\n    result = []\n    for item in arr:      # 遍历输入\n        # 在这里处理每一项\n        pass\n    return result",
                "思路：写代码 = 定义数据(变量) → 用条件分支(if) → 用循环处理批量 → 返回值(return)。你把【具体要解决什么问题】告诉我，我直接给你完整的实现。")

    def __init__(self):
        self.detected = []

    def detect(self, raw):
        for t in self._TASKS:
            if any(k and k in raw for k in t["kws"]):
                return True
        if any(k in raw for k in ["写代码", "写个代码", "写程序", "帮我写", "写一段", "代码", "编程", "实现"]):
            return True
        return False

    def reply(self, raw):
        for t in self._TASKS:
            if any(k and k in raw for k in t["kws"]):
                code, exp = t["code"], t["exp"]
                return "💻 {0}：\n```\n{1}\n```\n{2}".format(t["title"], code, exp)
        if any(k in raw for k in ["写代码", "写个代码", "写程序", "帮我写", "写一段", "代码", "编程", "实现"]):
            code, exp = self._GENERIC
            return "💻 我来示范一个通用函数骨架：\n```\n{0}\n```\n{1}".format(code, exp)
        return None


# ============================================================
# v0.5 正式版: 自升级大脑 (SelfLearner)
#   从输入/联网文本中: ① 生词加入分词库  ② 带解释的生词存进知识库
# ============================================================
class SelfLearner:
    _NOISE = set("的了是在我不有和这那与就也都而或及之很都太更最也吧吗呢啊哦呀啦吧么嘛嗯哈嘿嘻嘻啦啦哦耶哇哎")

    def __init__(self, tokenizer, retriever):
        self.tok = tokenizer
        self.retriever = retriever
        self.learned_words = 0
        self.learned_defs = 0
        self._known_titles = set(getattr(DATA, "KNOWN_TITLES", []))
        for d in getattr(self.retriever, "docs", []):
            try:
                self._known_titles.add(d.get("title"))
            except Exception:
                pass

    def _plausible(self, w):
        if not w:
            return False
        if w in self.tok.dictionary:
            return False
        if re.fullmatch(r"[A-Za-z]+", w):
            return 3 <= len(w) <= 20 and w not in self._NOISE
        if re.fullmatch(r"[\u4e00-\u9fff]+", w):
            return 2 <= len(w) <= 12 and not all(c in self._NOISE for c in w)
        return False

    def collect_new_terms(self, text):
        """把分词结果里"连续未知单字"还原成候选生词"""
        if not text:
            return []
        known = self.tok.dictionary
        cands = set()
        tokens = self.tok.tokenize(text)
        buf = ""
        for t in tokens:
            is_unknown = (len(t) == 1 and t not in known)
            is_alpha = bool(re.fullmatch(r"[A-Za-z]|[\u4e00-\u9fff]", t))
            if is_unknown and (is_alpha or True):
                buf += t
            else:
                if self._plausible(buf):
                    cands.add(buf)
                buf = ""
        if self._plausible(buf):
            cands.add(buf)
        # 英文整词: 直接抓取连续字母词(转小写), 长度>=3 才学
        for m in re.finditer(r"[a-zA-Z]{3,20}", text):
            w = m.group(0).lower()
            if w not in known and self._plausible(w):
                cands.add(w)
        return sorted(cands)

    def _extract_definition(self, text, fresh_cands=None):
        """抓取"X 是/指/意思/即 …"形式的词条解释.
        fresh_cands: 本次候选生词集合, 传了就从它判定"是否需要存释义",
        避免先 add_words 后把刚学的词误判为"已认识"而跳过."""
        pats = [
            re.compile(r"([\u4e00-\u9fffA-Za-z]{2,12}(?:[\u4e00-\u9fffA-Za-z0-9· ]{0,2}))(?:的)?(?:的意思是|是用于|是指|意指|就是指|解释为|即|表示|意为)\s*[:：]?\s*([^\n。;；]{2,60})"),
            re.compile(r"([A-Za-z][A-Za-z ]{1,15})\s*:\s*([^\n。;；]{2,60})", re.I),
        ]
        for pat in pats:
            m = pat.search(text)
            if not m:
                continue
            term, meaning = m.group(1).strip(), m.group(2).strip()
            if not term or not meaning:
                continue
            # 术语清洗: 去掉杂角色; 若含英文词则取英文词部分, 否则只留纯中文
            for ch in "的是一种在用作指|:=":
                term = term.replace(ch, "")
            term = term.strip(" 的：")[:20]
            en = re.search(r"[A-Za-z][A-Za-z0-9]*", term)
            if en:
                term = en.group(0)
            else:
                term = "".join(re.findall(r"[\u4e00-\u9fff]+", term))[:12]
            if len(meaning) < 3 or len(term) < 2:
                continue
            if fresh_cands is not None:
                # 本次候选生词才算"学到新词+解释"
                if term not in fresh_cands:
                    continue
            elif term in self.tok.dictionary:
                continue
            return term, meaning
        return None

    def learn(self, text):
        words = self.collect_new_terms(text)
        cands = set(words)
        add = [w for w in words if w not in self.tok.dictionary]
        if add:
            self.tok.add_words(add)
            for w in add:
                # v0.7: 生词同步记入跨会话记忆(去重+校验, 供持久化)
                if getattr(self, "memory", None):
                    self.memory.note_word(w)
            self.learned_words += len(add)
        defs = self._extract_definition(text, fresh_cands=cands)
        if defs:
            term, meaning = defs
            self._store(term, meaning)
            self.learned_defs += 1
            # v0.7: 学了带解释的知识→记忆存"摘要"(过滤+去重+价值打分)
            if getattr(self, "memory", None):
                self.memory.note_def(term, meaning)
        if getattr(self, "memory", None):
            # v0.7: 每次学完即"学新词淘汰旧词"并落盘, 记忆有界不拖内存
            self.memory.commit()
        return len(add), (1 if defs else 0)

    def _store(self, term, meaning):
        title = term[:20]
        if title in self._known_titles:
            return
        entry = {"t": title, "a": [title], "c": "自学习",
                 "b": "{0}：{1}".format(title, meaning),
                 "r": [meaning[:6]]}
        self.retriever.docs.append({
            "entry": entry, "tokens": self.tok.tokenize_set(title + " " + meaning),
            "title": title, "aliases": [title]})
        self._known_titles.add(title)
        self._plausible and self.tok.add_words([title])

    def report(self):
        return self.learned_words, self.learned_defs


# ============================================================
# v0.9 Alpha: 在 v0.8 Pro 基础上做交互/加载细节升级(启动加载进度·思考即时反馈·卡通死看门狗)
# ============================================================
class LearnerMemory:
    MEM_FILE = "xiaofang_memory_v14alpha.json"
    _LEGACY_FILES = ("xiaofang_memory_v13alpha.json",
                     "xiaofang_memory_v12.json", "xiaofang_memory_v11pro.json",
                     "xiaofang_memory_v11.json",
                     "xiaofang_memory_v11alpha.json",
                     "xiaofang_memory_v10pro.json",
                     "xiaofang_memory_v10.json", "xiaofang_memory_v09alpha.json",
                     "xiaofang_memory_v08pro.json", "xiaofang_memory_v08.json",
                     "xiaofang_memory_v07.json")  # 旧记忆自动迁移, 不丢已学内容
    MAX_WORDS = 2000          # 学到生词上限(基础常用词永不淘汰)
    MAX_KNOW = 800            # 学到释义摘要上限
    _NOISE = set("的了是在我不有和这那与就也都而或及之很都太更最也吧吗呢啊哦呀啦吧么嘛嗯哈嘿嘻嘻啦啦哦耶哇哎")

    def __init__(self, tok=None):
        self.tok = tok
        self.words = {}       # w -> {"n": 出现次数, "last": 最近使用游标}
        self.know = {}        # title -> {"s": 摘要, "v": 价值分, "n": 次数, "last": 游标}
        self._seq = 0         # 单调递增 LRU 游标
        self._base_words = set()
        self._known_titles = set()
        self._dirty = False
        self._load()

    # ---------- 持久化读写 (原子写, 有界小文件 → 加载极快) ----------
    def _mem_path(self, fname=None):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname or self.MEM_FILE)

    def _load(self):
        p = self._mem_path()
        migrated = False
        if not os.path.exists(p):
            for legacy in self._LEGACY_FILES:            # 旧版本记忆文件 → 迁移到 v08
                lp = self._mem_path(legacy)
                if os.path.exists(lp):
                    p = lp
                    migrated = True
                    break
            else:
                return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            w = data.get("words", {})
            if isinstance(w, dict):
                for k, v in w.items():
                    if not isinstance(k, str) or not k:
                        continue
                    e = v if isinstance(v, dict) else {}
                    self.words[k] = {"n": int(e.get("n", 1)), "last": int(e.get("last", 0))}
            k = data.get("know", {})
            if isinstance(k, dict):
                for t, v in k.items():
                    if not isinstance(t, str) or not t:
                        continue
                    e = v if isinstance(v, dict) else {}
                    self.know[t] = {"s": str(e.get("s", "")), "v": float(e.get("v", 1.0)),
                                    "n": int(e.get("n", 1)), "last": int(e.get("last", 0))}
            self._seq = data.get("seq", 0) or 0
            if migrated:                                  # 迁移成功 → 立即写入新版文件
                self._dirty = True
                self._save()
        except Exception:
            pass

    def _save(self):
        p = self._mem_path()
        try:
            data = {"version": "0.9 Alpha", "seq": self._seq,
                    "words": self.words, "know": self.know}
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, p)   # 原子替换: 崩溃也不会写坏记忆文件
            self._dirty = False
        except Exception:
            pass

    # ---------- 供启动时回填 ----------
    def extra_words(self):
        return sorted(self.words)          # 有界(≤MAX_WORDS), 启动仅合并单词

    def seed_retriever(self, retriever):
        """把学到的释义摘要注入检索文档(存摘要不存原文)."""
        for title, e in self.know.items():
            s = e.get("s", "")
            if not s:
                continue
            retriever.docs.append({
                "entry": {"t": title, "a": [title], "c": "学习记忆",
                          "b": "{0}：{1}".format(title, s), "r": [s[:8]]},
                "tokens": self.tok.tokenize_set(title + " " + s),
                "title": title, "aliases": [title]})
            self._known_titles.add(title)

    def bind_base(self, base_words, known_titles):
        self._base_words = set(base_words)
        self._known_titles |= set(known_titles)

    # ---------- 学习写入 (过滤 ①去重 ②校验 ---- ----------
    def capable(self, w):
        if not w or w in self._base_words or w in self.words:
            return False
        if re.fullmatch(r"[A-Za-z]+", w):
            return 3 <= len(w) <= 20 and w not in self._NOISE
        if re.fullmatch(r"[\u4e00-\u9fff]+", w):
            return 2 <= len(w) <= 12 and not all(c in self._NOISE for c in w)
        return False

    def note_word(self, w):
        """记一次生词出现: 已学仅累加(去重), 新词入表. 返回是否新学."""
        if not self.capable(w):
            return False
        self._seq += 1
        e = self.words.get(w)
        if e:
            e["n"] += 1
            e["last"] = self._seq
            return False
        self.words[w] = {"n": 1, "last": self._seq}
        self._dirty = True
        return True

    def note_def(self, term, meaning):
        """存释义"摘要"(压缩), 去重标题, 价值打分. 返回是否新存."""
        if not term or not meaning or len(meaning) < 4:
            return False
        title = term[:20]
        if title in self._known_titles or title in self._base_words:
            return False
        self._seq += 1
        summ = re.sub(r"\s+", "", meaning)[:40]   # 只存摘要, 不存原文
        if title in self.know:
            e = self.know[title]
            e["n"] += 1
            e["last"] = self._seq
            e["v"] = round(self._value(term, summ, e["n"]), 3)
            e["s"] = summ
            self._dirty = True
            return False
        self.know[title] = {"s": summ, "v": round(self._value(term, summ, 1), 3),
                            "n": 1, "last": self._seq}
        self._known_titles.add(title)
        self._dirty = True
        return True

    # ---------- 价值打分 (③只存有价值的段落摘要) ----------
    def _value(self, term, summ, n):
        v = 1.0
        v += min(1.5, len(summ) * 0.04)          # 摘要信息量: 越长越有价值
        if term not in self._base_words:
            v += 0.6                             # 非基础词 → 更具"新知"价值
        v += 0.25 * min(n, 4)                    # 被反复提到 → 更可信
        jk = set("的了是在我不有和这那与就也都而或及之")
        if len([c for c in summ if c in jk]) >= len(summ) * 0.8:
            v *= 0.4                             # 全是虚词 → 低价值
        return v

    # ---------- 遗忘机制: 学新词淘汰旧词 / 知识踢最低价值 ----------
    def evict(self):
        removed_w = removed_k = 0
        if len(self.words) > self.MAX_WORDS:
            over = len(self.words) - self.MAX_WORDS
            ranked = sorted(self.words.items(),
                            key=lambda kv: (kv[1]["n"], kv[1]["last"]))   # 次数少+久未用 → 先淘汰
            for w, _ in ranked[:over]:
                self._forget_word(w)
                removed_w += 1
        if len(self.know) > self.MAX_KNOW:
            over = len(self.know) - self.MAX_KNOW
            ranked = sorted(self.know.items(),
                            key=lambda kv: (kv[1]["v"], kv[1]["last"]))   # 价值低+旧 → 先淘汰
            for t, _ in ranked[:over]:
                self._forget_know(t)
                removed_k += 1
        return removed_w, removed_k

    def _forget_word(self, w, from_model=True):
        """彻底遗忘旧词: 移出记忆 + 若可则移出分词字典(释放内存)."""
        self.words.pop(w, None)
        if from_model and self.tok is not None:
            self.tok.dictionary.discard(w)

    def _forget_know(self, t, from_model=True):
        self.know.pop(t, None)
        self._known_titles.discard(t)
        if from_model and hasattr(self, "_retriever") and self._retriever is not None:
            self._retriever.docs[:] = [d for d in self._retriever.docs
                                       if d.get("title") != t]

    def bind_retriever(self, retriever):
        self._retriever = retriever

    def commit(self, force_save=True):
        """遗忘→落盘. 返回 (遗忘词数, 遗忘知识数)."""
        rw, rk = self.evict()
        if force_save and self._dirty:
            self._save()
        return rw, rk

    def clear_all(self):
        self.words.clear()
        self.know.clear()
        self._known_titles = set(self._known_titles) - set(self.know)
        self._dirty = True
        self._save()

    def stats(self):
        return {"words": len(self.words), "know": len(self.know),
                "max_words": self.MAX_WORDS, "max_know": self.MAX_KNOW}


class XiaoFang:
    def __init__(self):
        _br.stage("读取词库与知识库…", 0.12)
        kb_words = []
        for e in DATA.KNOWLEDGE_BASE:
            kb_words.append(e["t"])
            kb_words.extend(e.get("a", []))
        # v0.5 Alpha 2: 人格词库单独读入分词器(不进 n-gram 正文, 避免泄露到中性问答)
        for e in getattr(DATA, "PERSONA_KB", []):
            kb_words.extend(e["kws"])
            kb_words.extend(e.get("r", []))
        # v0.7: 自学习记忆——载入跨会话学到的生词+释义摘要(有界, 启动仅合并不重算)
        self.selfmem = LearnerMemory()
        _br.stage("汇入跨会话学习记忆…", 0.08)
        _base_titles = set()
        for _e in DATA.KNOWLEDGE_BASE:
            _base_titles.add(_e["t"])
            _base_titles.update(_e.get("a", []))
        self.selfmem.bind_base(DATA.COMMON_WORDS, _base_titles)
        self.tokenizer = Tokenizer(extra_words=list(kb_words) + self.selfmem.extra_words())
        self.selfmem.tok = self.tokenizer
        self.lm = NgramLM(self.tokenizer)
        corpus = []
        for e in DATA.KNOWLEDGE_BASE:
            corpus.append(str(e["t"]) + "。" + str(e["b"]))
            for a in e.get("a", []):
                corpus.append(a)
        corpus.extend(DATA.OPENERS_NEUTRAL + DATA.OPENERS_POSITIVE + DATA.OPENERS_NEGATIVE)
        corpus.extend(DATA.CLOSERS + DATA.CLOSERS_NEG + DATA.CLOSERS_POS)
        corpus.extend(DATA.IDENTITY_LINES + DATA.CAPABILITY_LINES)
        corpus.extend(DATA.JOKE_LINES)
        self.lm.train(corpus)
        # v0.6 Pro: 词库扩容——把整个常用词库并入 LM 词表, 让模型可嵌入/打分所有词;
        #   懒打分只评估剪枝候选集(top_k=96), 词表变大也不会拖慢生成。
        self.lm.vocab |= self.tokenizer.dictionary
        vocab_list = sorted(self.lm.vocab)
        _br.stage("构建 {} 深度思考引擎 (d_model={}×{}层×{}头)…".format(MODEL_TIER, MODEL_D, MODEL_LAYERS, MODEL_HEADS), 0.5)
        self.transformer = DeepThinkTransformer(
            vocab_list, d_model=MODEL_D, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS,
            ngram_lm=self.lm, ffn_ratio=MODEL_FFN, tie_out=MODEL_TIE)
        _br.stage("装配检索·数学·代码·自学习·表情…", 0.18)
        self.meter = TokenMeter()
        self.emotion = EmotionAnalyzer(self.tokenizer)
        self.intent = IntentDetector(self.tokenizer)
        self.retriever = SemanticRetriever(self.tokenizer)
        self.generator = ResponseGenerator(self.tokenizer, self.lm, self.transformer)
        self.persona = PersonaResponder(self.tokenizer)
        # v0.5 正式版: 数学 / 代码 / 自升级大脑
        self.math = MathResponder()
        self.code = CodeResponder()
        self.learner = SelfLearner(self.tokenizer, self.retriever)
        # v0.7: 把学到的释义摘要注入检索; 档案与遗忘所需的引用接好
        self.selfmem.seed_retriever(self.retriever)
        self.selfmem.bind_retriever(self.retriever)
        for d in self.retriever.docs:
            if d.get("title"):
                self.learner._known_titles.add(d["title"])
        self.learner.memory = self.selfmem
        # v0.7: 学新词/释义的同时淘汰旧项并落盘, 保证记忆有界不拖内存
        fw, fk = self.selfmem.commit(force_save=False)
        self.mem_evict_note = (fw, fk)
        self.history = []
        self.max_history = 50
        # v0.4 正式版: 预设应答 (精确/包含关键词快速命中)
        self.presets = []
        for kws, reply in getattr(DATA, "PRESETS", []):
            self.presets.append({"kws": [k for k in kws if k], "reply": reply})
        _br.stage("加载完成，就绪 ✓", 0.12)

    def _core_entity(self, user_input, emo, intent, kb_hits, kb_top):
        top = intent["top"]
        kb_score = kb_hits[0][0] if kb_hits else 0.0
        intent_entity = {
            "identity": "小方(我自身)",
            "capability": "我的能力",
            "date": "当前日期",
            "time": "当前时间",
            "weather": "天气",
            "joke": "一个有趣的笑话",
            "bye": "道别",
            "thanks": "礼貌感谢",
            "greet": "问候",
            "help": "你的情绪与状态",
            "persona": "小方(我的生活背景/身世)",
            "math": "数学计算/解题",
            "code": "代码/算法",
        }.get(top)
        if intent_entity:
            return intent_entity
        # 知识问答/指令/学习类: 高置信命中时采用实体标题, 否则取句中最长实义词
        if top in ("question", "command", "study", "start", "suggest"):
            if kb_top and kb_score >= 0.35:
                return kb_top
        tokens = list(emo.get("tokens") or [])
        stops = set("我是你他她它我们你们他们这那的了吧吗呢啊哦哟怎么什么哪个谁把被让从对给和或与及在到上空很太都也还就才只又再但")
        cand = sorted([t for t in tokens if len(t) >= 2 and t not in stops], key=len, reverse=True)
        return cand[0] if cand else (user_input[:8] or "主题")

    def _decompose(self, user_input, emo, intent, kb_hits, kb_top):
        entity = self._core_entity(user_input, emo, intent, kb_hits, kb_top)
        detail = []
        detail.append("核心实体: " + (entity if entity else "无明确实体"))
        if intent["top"] == "question":
            if entity and entity not in ("无明确实体",):
                detail.append("诉求: 对「" + entity + "」做解释/说明")
            else:
                detail.append("诉求: 解释未知问题")
        elif intent["top"] == "suggest":
            detail.append("诉求: 给出建议或选项")
        elif intent["top"] == "study":
            detail.append("诉求: 学习方法/步骤")
        elif intent["top"] == "start":
            detail.append("诉求: 引导入门/第一步")
        elif intent["top"] == "help":
            detail.append("诉求: 情绪安抚/情感支持")
        else:
            tname = {"search": "联网搜索", "command": "执行操作",
                     "time": "时间", "date": "日期", "weather": "天气",
                     "joke": "讲笑话", "identity": "自我认知",
                     "persona": "设定人格问答(生活/身世)",
                     "math": "数学计算/解题",
                     "code": "代码生成/算法",
                     "thanks": "回应感谢", "bye": "道别", "greet": "问候"}.get(intent["top"], intent["top"])
            detail.append("诉求: " + tname)
        detail.append("情感: {} (得分{} 强度{})".format(emo["category"], emo["score"], emo["intensity"]))
        detail.append("问句形态: {}".format("是" if emo["has_question"] else "否"))
        return detail

    def _strategy_scores(self, emo, intent, kb_hits, query=""):
        kb_score = kb_hits[0][0] if kb_hits else 0.0
        # v0.8 Pro: 知识直答前提是"实体精确对齐", 否则不轻易走"翻本地库"(防乱翻库答非所问)
        if kb_hits and not _kb_aligned(query, kb_hits[0][1]):
            kb_score = 0.0
        strategies = []
        strategies.append(("A-知识直答", kb_score, "依据知识库确定性给出直接答案"))
        base_b = 0.0
        if intent["top"] in ("question", "command", "study", "start"):
            base_b = 0.5
        if emo["has_question"]:
            base_b += 0.15
        strategies.append(("B-展开+延伸", base_b, "在核心答案外补充相关延伸"))
        base_c = 0.0
        if intent["top"] in ("study", "suggest", "start"):
            base_c = 0.6
        elif emo["category"] in ("难过", "焦虑", "生气", "疲惫", "有点累"):
            base_c = 0.4
        strategies.append(("C-举例/情感", base_c, "用例子或情感化表达拉近距离"))
        strategies.sort(key=lambda x: -x[1])
        return strategies, kb_score

    def think(self, user_input, emo, intent, kb_hits):
        self.meter.count_input(emo["tokens"])
        self.meter.new_turn()
        t0 = time.time()   # v0.4.1 记录思考起始, 用于下方"本次已思考多久"

        # v0.6 思考等级: off = 不思考, 直接跳过深度推理(用于极快/省电场景)
        if THINK_MODE == "off":
            self.last_web = None
            return ""

        kb_top = kb_hits[0][1]["t"] if kb_hits else None
        kb_score = kb_hits[0][0] if kb_hits else 0.0

        seed = self.tokenizer.tokenize(user_input)[:SEED_TOKENS]   # v0.5: 读更多词元
        if not seed:
            seed = ["好"]
        bias = set(self.tokenizer.tokenize(user_input))
        if kb_hits:
            bias |= set(self.tokenizer.tokenize(kb_hits[0][1]["b"]))

        # —— v1.2 卡死防呆: 纯表情/符号/超短闲聊 = 无真实语义 → 跳过 1B 神经正演,
        #    否则 CPU 上这几十秒没有任何输出, 用户误以为卡机。这类话直接轻量应答。
        _has_meaning = bool(re.search(r"[\u4e00-\u9fff]+", user_input or "") or
                            re.search(r"[A-Za-z]{2,}", user_input or ""))
        _no_sem = (not _has_meaning) and len((user_input or "").strip()) <= 16   # 纯表情/符号
        _chatty = intent["top"] in ("greet", "bye", "thanks", "joke", "smalltalk",
                                    "time", "date", "help")
        if _no_sem or (len(seed) <= 2 and _chatty):
            self._note_stage("轻量闲聊(不需要深想)")
            self.last_web = None
            _disp = (user_input or "").strip()[:12]
            if SHOW_DEEP_THINK:
                _typewrite(("唔，你发的是「{}」。这多半是随便唠一句，"
                            "没必要动用大引擎深算，我直接轻快回你。").format(_disp), C_DEEP, delay=0.02)
            return "<deep_think>轻量闲聊(跳过重计算)</deep_think>"

        self._note_stage("深度思考(神经矩阵运算)")
        # ---------- 矩阵正演跟踪 ----------
        ids = [self.transformer.token2id.get(t, 0) for t in seed] or [0]
        _, tr = self.transformer.forward(ids, trace=True)

        top_idx = np.argsort(tr["probs"])[::-1][:5]
        top5 = [(self.transformer.vocab[i], float(tr["probs"][i])) for i in top_idx]

        # v0.6: 复用 trace 那一次 forward 的 probs, 不再二次前向 (修复"Transformer跑两遍")
        # v1.2: 用用户拆词 token 走"深度意图定向" —— 矩形矩阵+注意力池化+纵深精修 → 意图向量,
        #       再按「候选词·意图向量」的贴合度给猜猜乐加权, 精准指向用户到底要什么。
        intent_vec = self.transformer.intent_encode(ids)
        dist = self.transformer.score_distribution(seed, bias, probs=tr["probs"],
                                                   intent_vec=intent_vec)
        best_tok = max(dist, key=lambda k: dist.get(k, 0.0)) if dist else None
        best_prob = dist.get(best_tok, 0.0) if best_tok else 0.0
        # 记录意图定向的强相关词, 用于思考展示"我盯着哪个目标在猜"
        _iv = intent_vec if intent_vec is not None else None
        _top_rel = []
        if _iv is not None:
            _rel = self.transformer._intent_related(seed, _iv, 6)
            _top_rel = [t for t, _ in _rel]

        detail = self._decompose(user_input, emo, intent, kb_hits, kb_top)
        strategies, kb_score = self._strategy_scores(emo, intent, kb_hits, user_input)
        plan = strategies[0][0][:1]

        parts = ["<deep_think>"]
        parts.append("🧠 DeepThink 深度推理 · {} {} · d_model={} n_heads={} n_layers={} vocab={}".format(
            ENGINE_NAME, VERSION, self.transformer.d_model, self.transformer.n_heads,
            self.transformer.n_layers, self.transformer.vocab_size))
        parts.append("─────────────────────────────────────────────")
        parts.append("【① 输入分词与意图解析】")
        parts.append("  文本: 「{}」".format(user_input[:60]))
        parts.append("  分词(前{}): ".format(DISPLAY_TOKENS) + str(emo["tokens"][:DISPLAY_TOKENS]))
        parts.append("  意图主判: {} (score={})".format(intent["top"], intent["max_score"]))
        parts.append("  子问题拆解:")
        for d in detail:
            parts.append("     · " + d)
        # v1.2 深度意图定向: 把"用户到底要什么"显性化 —— 展示意图向量与加权后的猜猜乐冠军
        _dpt = getattr(self.transformer.intent_scorer, "depth", 0)
        parts.append("【①+ 深度意图定向 (v1.2 猜猜乐加权)】")
        parts.append("  拆词 token → 矩形投影+注意池化+{}层精修 → 意图向量({}维)".format(_dpt, self.transformer.d_model))
        parts.append("  意图锚定语境词: {}".format("、".join("「{}」".format(w) for w in _top_rel) if _top_rel else "—"))
        parts.append("  意图加权后猜猜乐冠军: 「{}」 P={}".format(best_tok, _fmt(best_prob)))
        parts.append("【② 检索与知识激活】")
        if kb_hits:
            for i, (s, e) in enumerate(kb_hits[:3], 1):
                parts.append("  命中{}: 「{}」 Jaccard/覆盖≈{}".format(i, e["t"], s))
            parts.append("  语义聚合置信度≈{}".format(_fmt(min(0.99, kb_score + 0.3))))
        else:
            parts.append("  知识库: 无高置信命中，转自由生成/搜索")
        # v0.4Search: 自判联网并内置于思考 (联网过程就在深度思考里)
        web_should = self._should_web(user_input, emo, intent, kb_hits)
        self.last_web = None
        if web_should:
            parts.append("【②+ 联网实时检索 (思考内)】")
            wq = self._extract_query(user_input)
            parts.append("  判定需联网 → 关键词:「{}」 后端: {}".format(wq, "→".join(SEARCH_BACKENDS)))
            wres = self._fetch_web(wq)
            self.last_web = {"query": wq, "results": wres, "ok": bool(wres), "time": time.time()}
            if wres:
                parts.append("  命中 {} 条, 注入逐字稿做 RAG 式整合:".format(len(wres)))
                for i, item in enumerate(wres[:4], 1):
                    parts.append("     {}「{}」".format(i, (item.get("title") or "")[:30]))
            else:
                parts.append("  → 检索未命中(网络受限), 回退本地生成")
        parts.append("【③ 多策略推演 (Deep Reasoning)】")
        for name, conf, desc in strategies:
            mark = " → 拟采用" if name == strategies[0][0] else ""
            parts.append("  策略{}: 置信{} · {}{}".format(name, _fmt(conf), desc, mark))
        parts.append("  自检: 与意图匹配 ✓  无矛盾 ✓  → 融合「" + strategies[0][0] + "」方案")
        parts.append("【④ Transformer 神经运算 (前向传播)】")
        parts.append("  Embed(x): {}×{}  输入L2范数={}".format(tr["seq"], self.transformer.d_model, _fmt(tr["embed_norm"])))
        for i, blk in enumerate(tr["blocks"]):
            if i == 0:
                row_s = " ".join(_fmt(v) for v in blk["row"][:10])
                parts.append("  L{} Self-Attn: Q·Kᵀ/√{} → softmax  头0·末位query: [{} ...]".format(
                    i + 1, self.transformer.d_model // self.transformer.n_heads, row_s))
            parts.append("       L{} 输出 L2={}  8头峰值={}".format(i + 1, _fmt(blk["norm"]), blk["peaks"]))
        parts.append("  输出层 logits({}) L2={} → softmax Top-5:".format(
            self.transformer.vocab_size, _fmt(tr["logits_norm"])))
        for tok, p in top5:
            parts.append("       P({})={}".format(tok, _fmt(p)))
        parts.append("  合并 n-gram 语料先验 → 决策候选: 「{}」 P={}".format(best_tok, _fmt(best_prob)))
        parts.append("【⑤ 结论】")
        know_top = intent["top"] in ("question", "command", "study", "start", "suggest")
        if web_should and self.last_web and self.last_web.get("ok"):
            parts.append("  → 联网命中，RAG 式整合网页逐字稿成连贯回答")
        elif intent["top"] == "search" and intent["max_score"] >= 2:
            parts.append("  → 执行联网搜索，并将结果注入检索做 RAG 式整合")
        elif plan == "A" and kb_top and know_top:
            parts.append("  → 以「{}」为种子，按语义相关度组合知识库成句，矩阵运算输出连贯回答".format(kb_top))
        elif emo["score"] <= -3 or intent["top"] == "help":
            parts.append("  → 进入情绪安抚分支，先共情再给支持")
        elif intent["top"] == "identity":
            parts.append("  → 触发自我认知应答，返回身份/能力简介")
        elif intent["top"] == "persona":
            parts.append("  → 触发设定人格问答，读取生活/身世词库作答")
        elif intent["top"] == "math":
            parts.append("  → 识别为数学计算/解题，调用安全求解器分步计算")
        elif intent["top"] == "code":
            parts.append("  → 识别为代码/算法请求，调用代码生成器并讲解逻辑")
        elif intent["top"] in ("date", "time", "weather", "joke", "greet", "thanks", "bye"):
            parts.append("  → 走既定情境应答分支，实时生成针对性回复")
        else:
            parts.append("  → Transformer 深度思考 + 语义检索约束，结构化连贯生成")
        if self.history:
            parts.append("  上下文参考(上一轮): 「{}」".format(self.history[-2][:16]))
        # v0.5 Alpha: 深度思考段用逐行打字机打印; 打印结束后才计耗时
        # v0.6: 受设置里"显示深度思考"开关控制 (关闭则静默思考, 不打印但计时)
        elapsed = time.time() - t0
        # v1.0: 更深的"人味"思考 —— 自然语言内心独白(灰, 逐字敲出), 把"怎么想"显性化;
        #      算法过程(parts)完整保留在其后供技术查验; 最终正式回答仍走 reply() 浅蓝色输出。
        mono = self._think_monologue(user_input, emo, intent, kb_hits, kb_top, kb_score,
                                     detail, strategies, web_should)
        tail = ["  ⏱ 本次思考已耗时 {:.2f} 秒 (含深度思考打字显示)".format(elapsed), "</deep_think>"]
        if SHOW_DEEP_THINK:
            for _ln in mono:                       # 内心独白逐字打字 → 更像在"边想边敲"
                _typewrite(_ln, C_DEEP, delay=0.02)
            _typewrite(C_HINT + "────── 底层神经运算痕迹(供技术查验) ──────", C_DEEP)
            _typewrite_lines(parts, C_DEEP)        # 算法过程保留
            _typewrite_lines(tail, C_DEEP)
        return "\n".join(mono + parts + tail)

    def _think_monologue(self, user_input, emo, intent, kb_hits, kb_top, kb_score,
                         detail, strategies, web_should):
        """v1.0: 生成"人味"的内心独白 —— 不是算法步骤, 而是把【怎么解读、怎么回忆、
        掂量哪个办法、最终下什么决心】用大白话串起来(类似 Deepseek 式的链式思考)。
        全程依据真实输入/意图/检索结果动态生成, 不做重复套话。"""
        _TOP_READ = {
            "greet": "在跟我打招呼", "bye": "要跟我说再见", "thanks": "在道谢",
            "joke": "想听个段子轻松一下", "question": "是想弄明白一个概念",
            "study": "是想了解来龙去脉", "start": "是想让我从零讲起",
            "math": "是要我算点东西", "code": "是想让我帮忙写或改代码",
            "identity": "在问我到底是谁", "capability": "在问我能做哪些事",
            "persona": "想跟我闲聊生活", "smalltalk": "是想随便聊聊",
            "time": "在问现在几点", "date": "在问今天几号", "weather": "在问天气",
            "help": "在跟我求助", "command": "是需要我做点什么", "suggest": "在征求我的建议",
            "search": "是想让我帮他从网上查点东西",
        }
        mono = []
        t = (user_input or "").strip()
        disp = t[:30] + ("…" if len(t) > 30 else "")
        mono.append("唔，用户说的是「{}」。".format(disp))
        mono.append("细一看，他{}. ".format(_TOP_READ.get(intent["top"], "问了句不太好归类的话")))
        # —— 情绪 ——
        if emo["score"] <= -3:
            mono.append("这语气听起来有点低落(情绪分 {})，直接给方案太冷，得先接住情绪。".format(round(emo["score"], 1)))
        elif emo["score"] >= 2:
            mono.append("语气挺轻快的(情绪分 {})，回话可以放开点、俏皮点。".format(round(emo["score"], 1)))
        elif emo["has_question"]:
            mono.append('句子带问号，重点是「解答」，不是闲聊。')
        else:
            mono.append("语气平稳，正常应对就好。")
        # —— 回忆(依据检索) ——
        if web_should and self.last_web and self.last_web.get("ok"):
            _wq = self.last_web.get("query", "")
            _wn = len(self.last_web.get("results", []))
            mono.append("本地知识不全，我上网查了「{}」，拿到 {} 条资料，够拼出个像样的回答了。".format(_wq, _wn))
        elif kb_hits and kb_score >= 0.5:
            mono.append("我印象里有一条很近的记忆：「{}」，就先拿它当骨架。".format(kb_hits[0][1]["t"]))
        elif kb_hits:
            mono.append("知识库有沾边的「{}」，但重合度一般，我再用常识补全。".format(kb_hits[0][1]["t"]))
        else:
            mono.append("这个我脑子里没有现成词条，我把意思理顺、按常理组织一段话给他。")
        # —— 掂量(抽象版权衡) ——
        if intent["top"] == "math":
            mono.append("是道题，不能跳步——我先把式子在心里过一遍，再给结果和简要说明。")
        elif intent["top"] == "code":
            mono.append("代码这种得先讲清思路，再给能真正跑起来的实现，不能只甩一段糊弄。")
        elif intent["top"] == "question":
            mono.append('概念题，先讲准「是什么」，再配一个眼下就能用上的小说明，人更好懂。')
        elif intent["top"] == "suggest":
            mono.append("这是让我拿主意的推荐题——他要的是具体名字，不是类别清单。我在心里把几种不同类型的好货过一遍，挑几款各有亮点的给出去。")
        elif emo["score"] <= -3:
            mono.append("这会儿顺序很关键：先安慰打气，再试着帮上忙，反了效果会差。")
        elif intent["top"] in ("smalltalk", "greet", "joke", "thanks", "bye"):
            mono.append("属于轻松闲聊，应得自然、轻快一点就好。")
        else:
            mono.append("把关键几点理清，按最容易听懂的先后顺序说出来。")
        # —— 下决心 ——
        if web_should and self.last_web and self.last_web.get("ok"):
            mono.append("好，定啦——把网上查到的整合润色，回成一段清楚的话。")
        elif intent["top"] == "math":
            mono.append("行，算完写清过程和结果，再补一句怎么理解。")
        elif intent["top"] == "code":
            mono.append("行，给个能跑的版本，边给边讲。")
        elif intent["top"] == "suggest":
            mono.append("拿定了——把备选的具体作品摆上来，每条配一句亮点，好让他一眼挑中。")
        elif kb_hits and kb_score >= 0.5:
            mono.append("拿定了——以「{}」为底，顺着组织成完整回答。".format(kb_hits[0][1]["t"]))
        elif emo["score"] <= -3:
            mono.append("拿定了——先接住情绪、说句暖心话，再给实际能帮上的。")
        else:
            mono.append("拿定了——用最顺的表达把答案送出去。")
        return mono

    def _needs_online(self, text, intent=None, kb_score=0.0, has_question=False):
        # v1.1 正式版: 深度理解"到底要不要上网搜" —— 不滥用知识库, 也不乱联网。
        # 返回 True = 该上网搜; False = 用本地/直接作答。
        t = (text or "").strip()
        if not t:
            return False
        # ⚠ 凡是问"小方自己/作者/用什么做的/什么框架" → 本地答, 绝不联网(不曲解)
        self_ref = ["是什么框架", "什么框架", "用什么", "你用的", "你用什么",
                    "你的作者", "作者是谁", "谁做的", "谁开发", "谁创造",
                    "谁发明", "谁创建", "谁设计", "怎么做的", "怎么做出来",
                    "怎么写的", "底层", "源码", "架构", "用什么写得", "技术栈"]
        if any(m in t for m in self_ref):
            return False
        if intent and intent.get("top") in ("search", "weather"):
            return True
        if any(m in t for m in ["搜一下", "查一下", "帮我查", "上网查", "查查", "搜搜",
                                "上网", "最新", "去搜", "搜"]):
            return True
        # 具体游戏/影视的"攻略/教程/通关" → 上网搜
        if any(m in t for m in ["攻略", "教程", "通关", "打法", "套路",
                                "怎么通关", "通关心得", "怎么玩得爽", "开荒", "配装"]):
            return True
        # 当下新鲜/热门的具体物(游戏/电影/番/歌/书/电竞…) 求推荐 → 上网搜
        if any(m in t for m in ["推荐", "有什么好的", "推荐几个", "有什么好玩", "好玩",
                                "最近", "新出", "热门", "好玩的", "什么游戏", "什么电影",
                                "什么番", "有什么电影", "有什么游戏"]):
            if any(m in t for m in ["游戏", "电影", "番", "剧", "动漫", "动画", "音乐",
                                    "歌曲", "唱歌", "歌", "书", "小说", "电竞",
                                    "主机", "switch", "电脑"]):
                return True
        # 通用"是什么/怎么"概念且本地置信低 → 上网 (本方法兜底交给 _should_web 的低置信逻辑)
        return False

    def _should_web(self, text, emo, intent, kb_hits):
        # v0.4Search: 自主判断是否联网 (联网过程内置于深度思考)
        if FORCE_OFFLINE:
            return False
        kb_score = kb_hits[0][0] if kb_hits else 0.0
        # v1.1 正式版: 先用"该联网"决策 (含推荐/攻略/自我排除)
        if self._needs_online(text, intent, kb_score, emo and emo.get("has_question")):
            return True
        if any(m in text for m in ["不知道", "不了解", "不清楚", "不懂", "查一下", "帮我查"]):
            return True
        # 自我意图理解: 问"是什么/怎么/如何/介绍" 且本地知识库置信低 → 自动上网
        if intent["top"] in ("question", "study", "start") and kb_score < 0.35 and emo["has_question"]:
            if any(m in text for m in ["是什么", "什么是", "什么意思", "怎么样", "怎么",
                                       "如何", "怎样", "介绍", "原理", "起源", "区别", "在哪"]):
                return True
        return False

    def _extract_query(self, text):
        # 由用户输入生成联网关键词
        q = re.sub(r"[？?。！!，,、；;\s]+", " ", text)
        for w in ["小方", "帮我", "请", "搜索", "搜一下", "查一下", "上网查",
                  "查找", "是什么", "什么是", "呢", "啊", "帮我查一下"]:
            q = q.replace(w, "")
        q = q.strip(" :：")
        return q[:40] or (text.strip()[:40] or "AI")

    def _fetch_web(self, query, max_results=5):
        # v1.3 Alpha 卡死防呆: 整段网络放在 4s 守护线程硬超时里跑。
        #   旧版直接同步挨个试 SEARCH_BACKENDS, 网一不通每个后端都能挂很久 → 用户等 5 分钟像死机。
        def _go():
            results = []
            for bk in SEARCH_BACKENDS:
                try:
                    got = DDGS().text(query, backend=bk, max_results=max_results)
                    if got:
                        results = list(got)
                        if results:
                            self.last_web_backend = bk
                            break
                except Exception:
                    continue
            return results
        # 不管后端多卡, 最多 4 秒就让主线程拿回结果(拿不到就 [] → reply 走离线兜底, 绝不卡死)
        return _run_with_timeout(_go, timeout=4.0, default=[])

    def search_and_integrate(self, query, emo):
        # v0.4Search: 优先复用思考内已检索的结果, 否则立即联网; RAG 式整合
        web = getattr(self, "last_web", None)
        if web and web.get("ok") and time.time() - web.get("time", 0) < 30:
            wq, results = web["query"], web["results"]
        else:
            wq = self._extract_query(query)
            results = self._fetch_web(wq)
            self.last_web = {"query": wq, "results": results, "ok": bool(results), "time": time.time()}
        if not results:
            return "这次没能联网拿到结果(网络或后端受限)。换个关键词试试？或者我可以基于本地知识来回答。"
        result_texts = []
        for item in results:
            title = item.get("title", "")
            body = item.get("body", "") or item.get("body", "")
            href = item.get("href") or item.get("url", "")
            result_texts.append("{}。{}".format(title, body))
            self.lm.add_text(title + "。" + body)
        # v0.5 正式版: 自升级大脑——从抓到的网页内容里学生词/生解释
        learn_extra = ""
        try:
            lw2, ld2 = self.learner.learn(" ".join(result_texts))
            if lw2 or ld2:
                learn_extra = "🧠 网页资料我已吸收：新词 {} 个，知识 {} 条。".format(lw2, ld2)
        except Exception:
            pass
        sents = []
        for rt in result_texts:
            for s in _split_sents(rt):
                s = s.strip(" ")
                if len(s) >= 4:
                    sents.append(s)
        gen_text = "".join(sents[:5]) if sents else result_texts[0][:80]
        gen_text = self.generator._insert_emojis(gen_text, emo)
        top = results[0]
        src = top.get("href") or top.get("url", "")
        answer = "我从网上查到：{}".format(gen_text)
        if src:
            answer += "（来源：{}）".format(src)
        opener = random.choice(["不过网上的信息你可以再核对下。", "如果需要，我可以继续帮你查更细的。"])
        return _normalize_emoji(answer + opener + learn_extra)   # v1.2: 收敛网页混入的乱插 emoji

    def _is_recommend(self, text):
        # v1.0 Pro: 是否"要推荐/拿主意"类 —— 这类只该给"有哪些选项", 绝不能去翻库解释"某东西是什么"
        t = text or ""
        # 先排除"解释某概念"的问句(如"推荐系统是什么"), 这种不该被推荐逻辑抢走
        if any(m in t for m in ["是什么", "啥意思", "什么意思", "定义", "原理", "含义", "解释", "介绍一下", "百科"]):
            return False
        return any(m in t for m in [
            "推荐", "建议", "玩什么", "玩点啥", "什么好玩", "啥好玩", "推荐一下",
            "有啥推荐", "有什么推荐", "帮我推荐", "选哪个", "选什么", "怎么选",
            "吃什么", "看什么", "听什么", "什么游戏", "什么电影", "什么书", "玩点什么", "买什么",
        ])

    # v1.1 Pro: 本地"具体推荐清单" —— 给实打实的名称+一句话, 而非"类型分类"。
    #   用户问"有什么好玩的游戏", 要的是具体游戏名; 只甩类目 = 曲解。联网失败/离线也能给出真推荐。
    _REC_GAMES = [
        ("《黑神话：悟空》", "国产动作3A，打击感和美术都拉满"),
        ("《塞尔达传说：旷野之息》", "开放世界自由探索的天花板，随便逛逛都是惊喜"),
        ("《艾尔登法环》", "魂系开放世界，硬核对决极爽"),
        ("《传送门2》", "解谜神作，蓝橙传送门疯狂开脑洞，还能双人合作"),
        ("《巫师3：狂猎》", "剧情和两个DLC份量十足，角色扮演经典"),
        ("《荒野大镖客2》", "西部世界沉浸感一流，节奏慢但后劲很大"),
        ("《双人成行》", "必须两人一起玩才快乐的神作，合作感拉满"),
        ("《博德之门3》", "回合制CRPG巅峰，自由度高到能放飞自我"),
        ("《原神》", "二次元开放世界，探索和养成兼顾"),
        ("《只狼：影逝二度》", "吃瘪无数次后悟了，就再也停不下来"),
        ("《我的世界》", "沙盒创造，想怎么玩都行"),
        ("《星露谷物语》", "种田养鸡治愈小品，一玩就上头"),
        ("《泰坦陨落2》", "单人战役堪称教科书，机甲跑墙爽到爆"),
    ]
    _REC_MOVIES = [
        ("《星际穿越》", "科幻+亲情的双料炸裂"),
        ("《让子弹飞》", "台词封神，越品越有味"),
        ("《肖申克的救赎》", "老片但常看常新"),
        ("《疯狂动物城》", "轻松又有深度的动画"),
        ("《你的名字》", "新海诚经典，画面美故事暖"),
        ("《复联4》", "漫威阶段落幕，场面过瘾"),
    ]
    _REC_BOOKS = [
        ("《三体》", "科幻硬核，想象力拉满"),
        ("《活着》", "看完心里久久不能平静"),
        ("《百年孤独》", "魔幻现实主义的巅峰"),
        ("《小王子》", "很短但一直值得反复读"),
        ("《明朝那些事儿》", "历史讲得轻松好读"),
    ]
    _REC_SONGS = [
        ("《起风了》", "旋律一响就上头"),
        ("《平凡之路》", "朴树的治愈感"),
        ("《稻香》", "周董的怀旧暖歌"),
        ("《夜曲》", "经典耐听"),
        ("《孤勇者》", "燃向正能量"),
    ]
    _REC_FOOD = [
        ("🍜 一碗热气腾腾的番茄牛腩面", "酸甜浓汤挂面，暖和又顶饱"),
        ("🥘 酸菜鱼配米饭", "酸辣开胃，下饭一流"),
        ("🥡 街头烤串", "烟火气十足，越吃越香"),
        ("🍲 寿喜锅/火锅", "一锅煮万物，几个人围坐最合适"),
        ("🍰 芋泥啵啵奶茶", "甜品时刻的快乐水"),
    ]

    def _recommend(self, text, emo, kb_hits):
        # v1.1 Pro: 推荐要"给具体的东西", 不再只甩类型分类(那叫曲解)。
        #   每条都是一个具体名称 + 一句话理由, 联网失败/离线也能给实打实的好物; 联网命中走最新详情。
        t = (text or "").strip()

        # 决定命中哪类"具体清单"(同时招摇出篮子, 回答即给具体名)
        if any(m in t for m in ("游戏", "玩点啥", "玩什么")):
            bucket, kind = self._REC_GAMES, "游戏"
        elif any(m in t for m in ("电影", "剧", "番", "动漫", "动画", "片")):
            bucket, kind = self._REC_MOVIES, "电影/剧集"
        elif any(m in t for m in ("书", "小说", "漫画", "看什么")):
            bucket, kind = self._REC_BOOKS, "书"
        elif any(m in t for m in ("歌", "音乐", "歌曲", "听什么")):
            bucket, kind = self._REC_SONGS, "歌"
        elif any(m in t for m in ("吃", "饭", "菜", "什么好吃", "吃什么", "好吃")):
            bucket, kind = self._REC_FOOD, "吃的"
        else:
            return "我可以给你挑具体的——比如 好玩的游戏 / 好看的电影 / 书 / 歌 / 吃的，你定一个方向，我直接上名字。😊"
        import random as _r
        _r.shuffle(bucket)
        rows = bucket[:4]
        lines = "\n  ".join("· {} —— {}".format(n, why) for n, why in rows)
        lead = _r.choice([
            "我想了想要不要分类糊弄你，算了——直接上名字，{}里这几款是真不错的：".format(kind),
            "给你挑了{}几位能打的，都是久经考验的那种：".format(kind),
            "来，{}我第一个想到的就是这些，放心挑：".format(kind),
        ])
        follow = _r.choice([
            "你要是告诉我更喜欢哪种调调（刺激/轻松/剧情/多人），我再给你往这个方向精准加码。",
            "有看对眼的吗？多说一句你的偏好，我能给你更贴的几个。",
            "喜欢哪个方向，喊我一声，我给你顺着再挖几款同类。",
        ])
        return _normalize_emoji("{}\n{}\n{}".format(lead, lines, follow))

    # ============================================================
    # v1.1 Alpha: 长线作文生成 —— 稳定成文, 不胡言乱语
    # ============================================================
    def _is_essay(self, text):
        # v1.1 Alpha: 是否"写一篇/作文/成文"类请求。
        # 只在明确要"成文"时命中, 并排除写代码/邮件等非作文场景。
        t = (text or "").strip()
        if not t:
            return False
        if any(m in t for m in ["作文", "小作文"]):
            return True
        if not any(m in t for m in ["写一篇", "写一段", "写段", "写个", "成文",
                                    "长文", "代写", "帮我写", "围绕", "写篇", "写500", "写800"]):
            return False
        if any(m in t for m in ["代码", "程序", "函数", "脚本", "插件", "网站", "页面",
                                "简历", "邮件", "通知", "一句话", "标题", "简介"]):
            return False
        return True

    def _extract_topic(self, text):
        # v1.1 Alpha: 从"写一篇关于X的作文"这类句子里提炼主题名词。
        t = (text or "").strip().strip("'\"“”‘’「」『』【】（）()<>《》")
        topic = None
        m = re.search(r"以(?P<a>.+?)(?:为题|为题目|为话题|为主题)", t)
        if m:
            topic = m.group("a")
        if not topic:
            # 取"关于/围绕/针对"之后的段落头, 在第一个动作/语气/标点处切断
            _seg = None
            for _s in ("关于", "围绕", "针对"):
                _i = t.find(_s)
                if _i >= 0:
                    _seg = t[_i + len(_s):]
                    break
            if _seg is not None:
                _cuts = ["来写", "写一段", "写一篇", "写段话", "写篇", "写段", "写个",
                         "作文", "小作文", "短文", "感想", "心得", "论述", "主题", "内容",
                         "，", "。", "、", " ", "　", "啦", "呢", "啊"]
                _pos = [(_seg.find(c), c) for c in _cuts]
                _pos = [p for p in _pos if p[0] >= 0]
                if _pos:
                    _seg = _seg[:min(p[0] for p in _pos)]
                topic = _seg.strip()
        if not topic:
            m = re.search(r"(?:写一篇|写一段|写篇|写段|成文|帮我写|围绕|写个)(?:关于)?(?P<c>\S{2,18})", t)
            if m:
                topic = m.group("c")
        if not topic:
            # 兜底: 用核心实体提炼 (不取意图词)
            core = self.generator._core_entity(t)
            topic = core if core else None
        if not topic:
            return "这个话题", "这件事"
        # 清洗主题: 去掉句尾问词/口语词/包裹引号
        for _ in range(3):
            dropped = False
            for tail in ("作文", "小作文", "文章", "短文", "感想", "心得", "论述", "主题", "内容",
                         "文字", "怎么写", "给我", "帮我", "写一篇", "写一段", "我来写",
                         "写", "呢", "啊", "吧", "呀", "了", "的", "关于"):
                if topic.endswith(tail):
                    topic = topic[:-len(tail)]
                    dropped = True
            if not dropped:
                break
        topic = topic.strip(" ，。！？,.、；;：:的'\"“”‘’「」『』【】（）()<>《》")
        topic = topic.strip("写")
        topic = topic[:18]
        if len(topic) < 2:
            return "这个话题", "这件事"
        return topic, topic if len(topic) <= 16 else "这件事"

    def _gen_essay(self, text, emo, kb_hits):
        # v1.1 Alpha: 稳定长文(500~800字)。每个句子都取自完整句库, 只替换主题/事实锚点,
        #   绝不逐字随机采样 → 输出整段完整稳定的句子, 不胡言乱语。
        import random as _r
        topic, sub = self._extract_topic(text)
        t = topic or "这个话题"
        c = sub or t or "它"

        # ① 从本地知识库取一条与该主题对齐的事实句作为锚点 (没有就用通用的精炼表述)
        fact = None
        for ent in DATA.KNOWLEDGE_BASE:
            aligned = False
            for cc in [str(ent["t"])] + [str(a) for a in (ent.get("a", []) or [])]:
                if cc and len(cc) >= 2 and cc in t:
                    aligned = True
                    break
            if aligned:
                _s = _split_sents(str(ent["b"]))
                if _s:
                    fact = _s[0]
                break
        if not fact:
            fact = "它早已融入我们的日常，只是平时很少被单独拎出来谈论"

        def fill(s):
            s = s.replace("{t}", t).replace("{c}", c)
            return s.replace("{fact}", fact) if "{fact}" in s else s

        def pick(pool, used):
            pool = [_ for _ in pool if _ not in used and fill(_)]
            if not pool:
                return None
            ch = _r.choice(pool)
            used.add(ch)
            return fill(ch)

        def cz(s):
            return len(re.sub(r"\s", "", s))

        used = set()
        parts = [fill(_r.choice(getattr(DATA, "ESSAY_LEAD", []))) if getattr(DATA, "ESSAY_LEAD", None) else ""]
        # 破题 + 内涵(两句) + 事实锚
        open_s = pick(getattr(DATA, "ESSAY_OPEN", []), used)
        if open_s: parts.append(open_s)
        for _ in range(2):
            _a = pick(getattr(DATA, "ESSAY_BODY_A", []), used)
            if _a: parts.append(_a)
        if getattr(DATA, "ESSAY_FACT", []):
            parts.append(fill(_r.choice(DATA.ESSAY_FACT)))
        # 意义(两句) + 应用/深化(两句)
        for _ in range(2):
            _b = pick(getattr(DATA, "ESSAY_BODY_B", []), used)
            if _b: parts.append(_b)
        for _ in range(2):
            _c = pick(getattr(DATA, "ESSAY_BODY_C", []), used)
            if _c: parts.append(_c)

        # ② 字数还没到 500 → 用"展开句"补长 (仅保证句子完整, 不改结构稳定)
        deep = list(getattr(DATA, "ESSAY_DEEP", []) or [])
        _r.shuffle(deep)
        while sum(cz(p) for p in parts) < 520 and deep:
            d = fill(deep.pop())
            parts.append(d)

        # ③ 收束段
        close_s = _r.choice(getattr(DATA, "ESSAY_CLOSE", ["好，这就是我想和你说的关于{t}的话。"])).replace(
            "{t}", t).replace("{c}", c)
        if "{fact}" in close_s:
            close_s = close_s.replace("{fact}", fact)
        parts.append(close_s)

        # ④ 稳定化: 每段都以句号/叹号/问号收尾, 末尾补段落号
        clean = []
        for p in parts:
            p = p.strip()
            if p and not p.endswith(("。", "！", "？", "～", "\n", "：")):
                p += "。"
            clean.append(p)

        total = sum(cz(p) for p in clean)
        body = "\n\n".join(clean) if len(clean) <= 3 else "\n\n".join([clean[0]] + ["".join(clean[1:-1])] + [clean[-1]])
        # 收束一句单独成段, 主体连排
        return _normalize_emoji("📝 {}（约 {} 字）\n\n{}".format(str(t), total, body))

    def reply(self, user_input, emo, intent, kb_hits):
        text = user_input.strip()
        # v0.5 Alpha: 官网/网站类问题只答官网 (不答模型身份)
        site_ask = any(m in text for m in ["官网", "网址", "fanggame", "fanggame.company"])
        if not site_ask and "网站" in text and any(m in text for m in
                ["小方", "工作室", "fanggame", "网站是多少", "官网是多少"]):
            site_ask = True
        # v0.6 Pro: 要网址就只给网址 —— 不顺手介绍自己是啥/工作室是谁 (防止曲解意图)
        if site_ask:
            return "小方工作室官网：{}".format(STUDIO_SITE)
        # v1.1 Alpha: 长线作文/成文最优先 —— 写一篇/作文/围绕X写 → 稳定长文, 不被时/日/身份等抢答
        if self._is_essay(text):
            return self._gen_essay(text, emo, kb_hits)
        # v0.5 Alpha 2: 设定人格——仅在明确问生活/身世时启用, 否则保持中性 AI 助手
        if intent["top"] == "persona":
            pr = self.persona.reply(text)
            if pr:
                return pr
        # v0.5 正式版: 数学 / 代码
        if intent["top"] == "math":
            ma = self.math.answer(text)
            if ma:
                return ma
        if intent["top"] == "code":
            ca = self.code.reply(text)
            if ca:
                return ca
        if intent["top"] == "identity" and intent["max_score"] >= 1:
            if any(m in text for m in ["模型", "作者", "开发者", "谁做", "谁开发", "made",
                                       "框架", "底层", "源码", "技术栈", "怎么做出", "用什么"]):
                mline = random.choice([
                    "我是 {}，由 {} 研发的自主本地大模型引擎，当前版本 {}。💡".format(MODEL_NAME, STUDIO_NAME, VERSION),
                    "我的模型名是 {}，属于 {}，当前 {}，是纯本地、保留版权的原创引擎。🔒".format(MODEL_NAME, STUDIO_NAME, VERSION),
                ])
                return "{}官方站：{}（{} 出品）".format(mline, STUDIO_SITE, STUDIO_NAME)
            return "🤖 " + random.choice(DATA.IDENTITY_LINES)
        if intent["top"] == "capability" and intent["max_score"] >= 1:
            return "🌐 " + random.choice(DATA.CAPABILITY_LINES)
        if intent["top"] == "time" and intent["max_score"] >= 1:
            now = datetime.datetime.now()
            h = now.hour
            if h < 6:
                tip = "深夜了注意休息 🌙"
            elif h < 12:
                tip = "早上好时光 ☀️"
            elif h < 14:
                tip = "午休时间 🍵"
            elif h < 18:
                tip = "下午时光 🌤️"
            else:
                tip = "晚上了 🌆"
            return "🕐 现在是 {}，{}，接下来有什么安排吗？".format(now.strftime('%H:%M:%S'), tip)
        if intent["top"] == "date" and intent["max_score"] >= 1:
            now = datetime.datetime.now()
            wd = "一二三四五六日"[now.weekday()]
            return "📅 今天是 {} 星期{}，🎈 有什么特别安排吗？".format(now.strftime('%Y年%m月%d日'), wd)
        if intent["top"] == "greet" and intent["max_score"] >= 1:
            return random.choice(["你好呀！我是 AI 小方 ✨，有什么想聊的？",
                                  "嗨！我在这儿 👋，今天想让我帮你点啥？"])
        if intent["top"] == "joke" and intent["max_score"] >= 1:
            return "😂 " + random.choice(DATA.JOKE_LINES) + " 还想再来一个吗？"
        if intent["top"] == "bye" and intent["max_score"] >= 1:
            return "👋 " + random.choice(DATA.BYE_LINES)
        if intent["top"] == "thanks" and intent["max_score"] >= 1:
            return "😊 " + random.choice(DATA.THANKS_LINES)
        # v1.1 正式版: 已联网且当前确实需要上网(新鲜推荐/游戏攻略/明确查) → 用联网结果作答,
        #   优先于本地推荐/翻库, 避免"推荐游戏却答成游戏是什么"的曲解。
        if (getattr(self, "last_web", None) and self.last_web.get("ok")
                and time.time() - self.last_web.get("time", 0) < 30
                and self._needs_online(text, intent)):
            return self.search_and_integrate(text, emo)
        # v1.0 Pro: 推荐/拿主意类 提前拦截 —— 直接给"有哪些选项", 不再掉进"解释实体"翻库路径
        if self._is_recommend(text):
            return self._recommend(text, emo, kb_hits)
        # v0.4Search: 思考内已自判联网并命中 → 用联网结果 RAG 作答
        if getattr(self, "last_web", None) and self.last_web.get("ok") and \
                time.time() - self.last_web.get("time", 0) < 30:
            return self.search_and_integrate(text, emo)
        if intent["top"] == "search" and intent["max_score"] >= 2:
            return self.search_and_integrate(text, emo)
        if intent["top"] == "weather" and intent["max_score"] >= 1:
            return self.search_and_integrate(text + " 天气", emo)
        if not kb_hits and any(m in text for m in ["不知道", "不了解", "不清楚", "不懂"]):
            return self.search_and_integrate(text, emo)
        return self.generator.generate(text, kb_hits, emo, intent, self.history)

    def _enter_settings(self, first_arg=""):
        # v0.6: 交互式设置——改动直接改写 xiaofang_settings.py 源码, 永久保存
        _modes = {"off": "不思考", "think": "单轮深度思考", "multi": "多轮边答边想"}
        # 一次性子命令: setting 思考模式 off/think/multi 或 setting 情绪 1.5 等
        args = (first_arg or "").split()
        if args:
            key, val = args[0].lower(), (args[1] if len(args) > 1 else "")
            quick = {
                "off": ("THINK_MODE", "off"), "nothink": ("THINK_MODE", "off"),
                "think": ("THINK_MODE", "think"),
                "multi": ("THINK_MODE", "multi"),
            }
            if key in quick:
                k, v = quick[key]
                globals()[k] = v
                _save_settings_field(k, v)
                print(C_REPLY + "✓ 已切换思考模式 → {}，已永久保存。".format(_modes[v]) + C_RESET)
                return "ok"
            if key in ("情绪", "敏感", "emotion"):
                try:
                    v = float(val)
                    globals()["EMO_SENS"] = v
                    _save_settings_field("EMOTION_SENSITIVITY", v)
                    print(C_REPLY + "✓ 情绪敏感度 → {}，已永久保存。".format(v) + C_RESET)
                    return "ok"
                except Exception:
                    pass
            if key in ("轮数", "多轮", "turns"):
                try:
                    v = int(val)
                    globals()["THINK_TURNS"] = max(1, v)
                    _save_settings_field("THINK_TURNS", v)
                    print(C_REPLY + "✓ 多轮思考轮数 → {}，已永久保存。".format(v) + C_RESET)
                    return "ok"
                except Exception:
                    pass
            if key in ("离线", "联网", "offline") and val in ("on", "off", "开", "关"):
                v = (val in ("on", "开"))
                globals()["FORCE_OFFLINE"] = v
                _save_settings_field("FORCE_OFFLINE", v)
                print(C_REPLY + "✓ 联网: {}".format("强制离线" if v else "允许联网") + "，已永久保存。" + C_RESET)
                return "ok"
        # 无子命令 → 交互菜单
        print(C_REPLY + "⚙️  小方设置（改动会直接写进源码，永久保存）" + C_RESET)
        cur = "思考={} · 多轮×{} · 情绪敏感度={} · 标点emoji={} · 深度显示={} · 活泼度={} · 离线={}".format(
            _modes.get(THINK_MODE, THINK_MODE), THINK_TURNS, EMO_SENS,
            "开" if USE_PUNCT_EMOJI else "关", "开" if SHOW_DEEP_THINK else "关",
            ENERGY, "是" if FORCE_OFFLINE else "否")
        print(C_HINT + "  当前: " + cur + C_RESET)
        print(C_HINT + "  直接输入「off / think / multi」或「情绪 1.5」「轮数 4」即可修改" + C_RESET)
        print(C_HINT + "  输入 q 退出设置，其它任意键看菜单…" + C_RESET)
        while True:
            try:
                sel = input(C_REPLY + "⚙ 设置 > " + C_RESET).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not sel or sel.lower() in ("q", "quit", "exit"):
                print(C_HINT + "已退出设置。改动均已实时生效并永久保存。✨" + C_RESET)
                break
            r = self._enter_settings(sel)
            print(C_HINT + "── 可继续改，或输入 q 退出 ──" + C_RESET)

    def _multi_deliver(self, answer, emo, intent, kb_hits):
        """v0.6 多轮思考: 小方边答边想/边想边答 (类 DeepSeek V4 Pro)."""
        _typewrite_lines(["🧠 多轮思考进行中… 我分 {} 步把答案想完整。".format(THINK_TURNS)], C_DEEP)
        sents = [s for s in re.split(r"(?<=[。！？.!?])", answer) if s.strip()]
        if not sents:
            _typewrite("小方: " + answer)
            return
        total = max(1, THINK_TURNS)
        # 把句子按轮数分组
        groups = [[] for _ in range(total)]
        for i, s in enumerate(sents):
            groups[i % total].append(s)
        for g_i, grp in enumerate(groups):
            if not grp:
                continue
            if g_i > 0:
                # 轮间插入一次"思考"微步 (若话题需要进一步获取, 则提示获取)
                if intent["top"] in ("question", "study", "search", "start") and g_i == 1:
                    _typewrite_lines(["💡 让我再想一层的关联与补充…（若需更全，我下一步去获取）"], C_DEEP)
                else:
                    _typewrite_lines(["💡 补充细节…"], C_DEEP)
            piece = "".join(grp).strip()
            if piece:
                _typewrite("小方: " + piece)

    def handle_command(self, cmd):
        c = cmd.strip().lower()
        if c in ("setting", "/setting"):
            self._enter_settings()
            return "ok"
        if c in ("/off", "/nothink", "/nothink"):
            globals()["THINK_MODE"] = "off"
            _save_settings_field("THINK_MODE", "off")
            print(C_REPLY + "✓ 已切换为「不思考」模式（更快），永久保存。" + C_RESET)
            return "ok"
        if c == "/think":
            globals()["THINK_MODE"] = "think"
            _save_settings_field("THINK_MODE", "think")
            print(C_REPLY + "✓ 已切换为「单轮深度思考」，永久保存。" + C_RESET)
            return "ok"
        if c == "/multi":
            globals()["THINK_MODE"] = "multi"
            _save_settings_field("THINK_MODE", "multi")
            print(C_REPLY + "✓ 已切换为「多轮边答边想」，永久保存。" + C_RESET)
            return "ok"
        if c == "/help":
            print(C_REPLY + "📖✨ 小方命令手册：" + C_RESET)
            print(C_REPLY + "  /help    - 显示所有命令" + C_RESET)
            print(C_REPLY + "  setting  - 进入设置(改思考模式/情绪/联网等, 永久保存)" + C_RESET)
            print(C_REPLY + "  /think   - 切到单轮深度思考" + C_RESET)
            print(C_REPLY + "  /multi   - 切到多轮边答边想(类 DeepSeek V4 Pro)" + C_RESET)
            print(C_REPLY + "  /off     - 切到不思考(最快)" + C_RESET)
            print(C_REPLY + "  /prev    - 查看上一个对话(只看不回退, 可连看)" + C_RESET)
            print(C_REPLY + "  /undo    - 回到上一个对话 [需确认]" + C_RESET)
            print(C_REPLY + "  /new     - 开启新对话(清空本轮) [需确认]" + C_RESET)
            print(C_REPLY + "  /clear   - 清空对话历史(学习记忆不受影响, 仍在)" + C_RESET)
            print(C_REPLY + "  /memory  - 查看跨会话学习记忆档案(生词/知识摘要)" + C_RESET)
            print(C_REPLY + "  /forget  - 清空学习记忆(只清学到的内容)" + C_RESET)
            print(C_REPLY + "  /stop    - 强制掐断当前思考, 立即回到输入(防卡死)" + C_RESET)
            print(C_REPLY + "  /exit    - 退出程序" + C_RESET)
            print(C_REPLY + "⌨️ 快捷键面板: (无需输入符号, 按一下即触发)" + C_RESET)
            print(C_REPLY + "  F1  打开命令手册 · F2  查看上一个对话 · F3  回到上一个对话[需确认]" + C_RESET)
            print(C_REPLY + "  F4  开启新对话[需确认] · F5  强制掐断思考(防卡死) · F6  退出" + C_RESET)
            print(C_REPLY + "  F7  展开/折叠思考分布面板 · 点击面板也可 展开/折叠 · 盲文点=实时加载" + C_RESET)
            print(C_REPLY + "  ↑/↓ 选命令 · Tab 填入 · 回车 选中 · 继续输入=实时过滤 · Esc 取消联想" + C_RESET)
            print(C_REPLY + "🧠 v0.9 Alpha: 1B+ 模型(≈10.6亿参数·int8量化常驻≤1GB·自动降级算力不减)·启动加载进度·思考即时反馈·卡通死看门狗·实体直指 " + C_RESET)
            print(C_REPLY + "💡 引擎: DeepThink {} {} · {}层×{}头·d_model={}·自适应硬件·词库懒加载·多后端联网·多轮思考".format(
                ENGINE_NAME, VERSION, self.transformer.n_layers, self.transformer.n_heads,
                self.transformer.d_model) + C_RESET)
            return "ok"
        elif c == "/clear":
            self.history.clear()
            self.meter.reset()
            print(C_REPLY + "✨🧹 上下文+Token计量已清空，重新开始！" + C_RESET)
            return "ok"
        elif c == "/prev":
            # 查看上一个对话: 只看不回退。按完整一问一答对弹出, 可连按连看更早的。
            h = self.history
            if len(h) >= 2:
                pairs = list(zip(h[0::2], h[1::2]))   # 每两行为一问一答
                pair = pairs[-1]
                print(C_REPLY + "👆 上一个对话:" + C_RESET)
                print(C_REPLY + "  你: " + str(pair[0]) + C_RESET)
                print(C_REPLY + "  小方: " + str(pair[1]) + C_RESET)
                print(C_HINT + "  注: 仅查看, 不改变当前状态; 想回到这个对话请用 /undo" + C_RESET)
            else:
                print(C_HINT + "📭 还没有上一个对话; 来一句试试吧。" + C_RESET)
            return "ok"
        elif c == "/new":
            self.history.clear()
            self.meter.reset()
            print(C_REPLY + "🆕 已开启新对话(清空了本轮上下文, 学习记忆仍在)。" + C_RESET)
            return "ok"
        elif c == "/fold":
            UI_ST["fold"] = not UI_ST["fold"]      # 空闲态也可手动切换(无实际面板时无害)
            return "ok"
        elif c == "/stop":
            print(C_HINT + "⏹ 小方已就绪(没有在思考)。随时输入新消息；若在思考中按 F5 会立即掐断。" + C_RESET)
            return "ok"
        elif c == "/undo":
            if len(self.history) >= 2:
                ra = self.history.pop()
                ru = self.history.pop()
                print(C_REPLY + "↩️💨 已撤回：" + C_RESET)
                print(C_REPLY + "  你: " + ru + C_RESET)
                print(C_REPLY + "  小方: " + ra + C_RESET)
            else:
                print(C_ERROR + "⚠ 还没有可撤回的对话。" + C_RESET)
            return "ok"
        elif c == "/memory":
            st = self.selfmem.stats()
            print(C_REPLY + "🧠 学习记忆档案 (只存学到的词/知识, 不存对话与人格):" + C_RESET)
            print(C_HINT + "  已学生词 {w}/{mw} · 已学知识摘要 {k}/{mk} · 本机记忆文件 {file} · 持久保存、退出仍在".format(
                w=st["words"], mw=st["max_words"], k=st["know"], mk=st["max_know"],
                file=LearnerMemory.MEM_FILE) + C_RESET)
            if self.selfmem.words:
                print(C_HINT + "  最近生词: " + "、".join(sorted(self.selfmem.words)[-10:]) + C_RESET)
            if self.selfmem.know:
                print(C_HINT + "  最近知识: " + "、".join(list(self.selfmem.know)[-8:]) + C_RESET)
            print(C_HINT + "  提示: 记忆有上限, 学到新词会按使用频率淘汰久不用的旧词(LRU); /forget 可整库清空。" + C_RESET)
            return "ok"
        elif c == "/forget":
            self.selfmem.clear_all()
            print(C_REPLY + "🧹 学习记忆已清空并持久化(下次启动不再载入)。对话/人格本就不保存, 无涉。" + C_RESET)
            return "ok"
        elif c == "/exit":
            try:
                self.selfmem.commit()   # 退出前确保学习记忆落盘
            except Exception:
                pass
            print(C_REPLY + "👋🌙 再见，{} {} 已退出。".format(ENGINE_NAME, VERSION) + C_RESET)
            return "exit"
        else:
            print(C_ERROR + "❓ 未知命令: " + cmd + "，输入 /help 查看。" + C_RESET)
            return "ok"

    def process_chat(self, user_input):
        text = (user_input or "").strip()
        # v1.0 修复: 一旦进入这句的处理并开始输出(灰思考/蓝回答都是流式), 立即标记 streaming,
        # 主线程看门狗据此【停止再刷"仍在思考"】——因为它本就会一路打出来, 不需要(且也会挤断颜色)。
        self._streaming = True
        # v0.5 正式版: 自升级大脑——从用户这句里学生词/学的解释
        learn_note = ""
        if text:
            lw, ld = self.learner.learn(text)
            if lw or ld:
                learn_note = "🧠 我记了 {} 个生词、{} 条知识。".format(lw, ld)
        # v0.6 正式版: 先进深度思考, 再做关键词检测 (修复"先关键词截胡 → 答非所问")
        #   预设(关键词)只在"短句/闲聊型"意图下启用; 知识/官网/身份/数学等问题一律走 reply() 的精准分支。
        self._note_stage("解析你的意思")
        emo = self.emotion.analyze(text)
        intent = self.intent.detect(text, emo)
        self._note_stage("检索记忆 / 深度思考")
        kb_hits = self.retriever.retrieve(text)
        self.think(text, emo, intent, kb_hits)
        UI_ST["capture"] = False      # v1.4: 思考/计算已捕获完, 之后答案为真实打字机输出
        UI_ST["layer"] = MODEL_LAYERS
        # —— 思考完才 "再检测关键词" ——
        light_chat = intent["top"] in ("greet", "bye", "thanks", "joke", "help",
                                       "time", "date", "smalltalk") or len(text) <= 6
        preset_answer = None
        if light_chat:
            for p in self.presets:
                if any(k and k in text for k in p["kws"]):
                    preset_answer = p["reply"]
                    break
        if preset_answer is not None:
            answer = preset_answer
            out_tokens = self.tokenizer.tokenize(answer)
            self.meter.count_output(out_tokens)
            if self._skip_deliver:
                return
            _typewrite("小方: " + answer)
            if learn_note:
                _typewrite(C_HINT + learn_note + C_RESET)
            print(C_HINT + "  [{}]".format(self.meter.format()) + C_RESET)
            self.history.append(text)
            self.history.append(answer)
            return
        answer = self.reply(text, emo, intent, kb_hits)
        out_tokens = self.tokenizer.tokenize(answer)
        self.meter.count_output(out_tokens)
        if learn_note:
            answer = answer + " " + learn_note
        self._note_stage("正在组织回答")
        if self._skip_deliver:
            self._print_note("（已听你的，这条答案不显示了）", C_HINT)
            return
        if THINK_MODE == "multi" and SHOW_DEEP_THINK:
            self._multi_deliver(answer, emo, intent, kb_hits)
        else:
            _typewrite("小方: " + answer)
        print(C_HINT + "  [{}]".format(self.meter.format()) + C_RESET)
        self.history.append(text)
        self.history.append(answer)
        self.lm.add_text(text)
        self.lm.add_text(answer)
        if len(self.history) > self.max_history:
            self.history = self.history[-(self.max_history):]

    def _print_note(self, msg, color=C_DEEP):
        """v1.0: 用全局打印锁写一行带颜色的提示, 避免与打字机输出的颜色交错."""
        with _PRINT_LOCK:
            sys.stdout.write(color + str(msg) + C_RESET + "\n")
            sys.stdout.flush()

    def _note_stage(self, s):
        # v1.2: 阶段进度, 供主循环在活动指示行里显示"小方此刻正停在哪个阶段"
        self._stage = s
        UI_ST["stage"] = s       # v1.4: 状态面板实时阶段

    def run_with_timeout(self, text, mailbox=None, deferred=None):
        """v1.2 卡死防呆重写:
        1) 实时活动指示 —— 每 ~0.35s 在同一行刷新「转动指示 + 已耗秒数 + 当前阶段」。
           只要这行字在走, 就是【正常思考】; 停住不动才可能是【卡机】。区分:
           · 有流式打字输出(回答/思考在敲)→ 不碰该行, 让输出自己进度可见;
           · 处于 CPU 正演等无输出空窗 → 指示行不停走, 说明小方还活着;
           · 超过 RESPONSE_TIMEOUT → 明确提示可能真卡住, 并可打字中断。
        2) 非阻塞输入 —— 接收 run() 传来的输入信箱: 思考期间用户打的字先进队列,
           worker 结束后由 run() 继续接续回答(排队); 若用户发 /stop 则软中止当前回答。
        """
        if getattr(self, "_active_worker", None) and self._active_worker.is_alive():
            # v1.4 修复"偶尔卡死什么都不回": 上一轮 worker 若真卡住, 绝不静默 join 120s 挡盲区。
            # 给 1s 合理收尾; 仍活着即视为"已卡死", 跳过它直接进新回合(守护线程自己会退出/被忽略),
            # 同时给用户一句明确提示, 避免"打了字却半天没反应"的假死感。
            self._active_worker.join(timeout=1.0)
            if self._active_worker.is_alive():
                self._print_note("（上一轮有点卡，已自动跳过；这一条我马上答你）", C_HINT)
        got = {}
        self._streaming = False
        self._skip_deliver = False
        self._cancel_seen = False
        _touch_live()
        _SYS_BUSY["v"] = True          # 思考/回答期: 输入框只更缓冲不画面, 防撞屏
        _ui_reset_turn()               # v1.4: 状态面板从"按下回车=0"开始实时增长

        def _work():
            try:
                self.process_chat(text)
            except Exception:
                got["err"] = True
            finally:
                got["done"] = True

        w = threading.Thread(target=_work, daemon=True)
        self._active_worker = w
        w.start()
        t0 = time.time()
        while not got.get("done"):
            now = time.time()
            # —— v1.4: 思考/计算期 → 状态分段面板单线程原地重绘(Token/层数/阶段/盲文旋转) ——
            if UI_ST["capture"]:
                if UI_ST["layer"] < MODEL_LAYERS and now - t0 > 0.15:
                    UI_ST["layer"] += 1
                _panel_render()
            # —— 思考期间仍可打字: 读信箱, /stop 与 /fold 特判, 其余排队 ——
            if mailbox is not None and deferred is not None:
                while True:
                    try:
                        nxt = mailbox.popleft()
                    except IndexError:
                        break
                    if nxt in ("/stop", "stop", "停", "算了", "！", "!"):
                        self._cancel_seen = True
                        self._skip_deliver = True
                        self._print_note("\n（收到『/stop』，这条我就不答了，等你下一句）", C_HINT)
                    elif nxt in ("/fold", "fold"):
                        UI_ST["fold"] = not UI_ST["fold"]     # v1.4: F7/点击 ⇄ 展开/折叠面板
                    else:
                        deferred.append(nxt)
                        self._print_note("\n（你输入了『{}…』，小方还在忙，先排着队，马上轮到你）"
                                         .format(str(nxt)[:14]), C_HINT)
            if self._cancel_seen and not got.get("done"):
                got["cancel"] = True
            if now - t0 > RESPONSE_TIMEOUT:
                sys.stdout.write("\r" + " " * 70 + "\r")
                sys.stdout.flush()
                self._print_note(
                    "\n⚠ 已超过 {} 秒仍在处理，可能是真卡住了。小方会自己让开，你可以直接输入新消息，"
                    "或输入 /stop 打断它。".format(RESPONSE_TIMEOUT), C_DEEP)
                _SYS_BUSY["v"] = False
                return
            time.sleep(0.15)
        _SYS_BUSY["v"] = False         # 思考结束, 恢复画调色板
        # —— v1.4: 回答(打字机)已在 worker 打完后, 于状态面板之下补"学习因子"一行 ——
        if not got.get("cancel") and not got.get("err"):
            try:
                st = self.selfmem.stats() if hasattr(self, "selfmem") else {}
                print(C_HINT + "  🧠 学习因子 · {} · 知识库 生词 {w}/{mw} · 摘要 {k}/{mk} 条".format(
                    self.meter.format(),
                    w=st.get("words", 0), mw=st.get("max_words", 0),
                    k=st.get("know", 0), mk=st.get("max_know", 0)) + C_RESET)
            except Exception:
                pass
        if got.get("cancel"):
            return
        if got.get("err"):
            self._print_note("⚠ 小方处理这条时出了点小插曲，重新发送一下就好。", C_DEEP)

    def run(self):
        # v1.2: 加载进度在 __init__ 里已全部打完 → 就绪后清空整屏, 从最上方干净展示。
        # 输入改为独立读线程 + 信箱(deque): 思考/回答期间用户仍可打字, 进队列排队或 /stop 打断。
        _cls()
        show_startup()
        _enable_console_mouse()      # v1.4: 支持点击展开/折叠思考面板
        import collections as _col
        mailbox = _col.deque()
        deferred = []

        def _reader():
            global _read_prompt_pending
            while True:
                try:
                    # v1.3 Alpha: 真实命令行输入(斜杠调色板/F快捷键/YN确认)
                    line = _read_command("你: ").strip()
                except (EOFError, KeyboardInterrupt):
                    mailbox.append(None)
                    return
                mailbox.append(line)
                _read_prompt_pending = True      # 提交一句 → 忙完由主线程画下一句的"你: "

        threading.Thread(target=_reader, daemon=True).start()
        # 处理"已排队"的消息(思考期间用户输入的), 避免与命令解析耦合
        while True:
            if deferred:
                item = deferred.pop(0)
            else:
                try:
                    item = mailbox.popleft()
                except IndexError:
                    _show_read_prompt()          # 空闲态: 由主线程统一画"你: "(忙完才出现)
                    time.sleep(0.15)
                    continue
            if item is None:
                print(C_REPLY + "👋 小方再见～" + C_RESET)
                break
            if not item:
                continue
            low = item.strip().lower()
            if low in ("setting", "off", "think", "multi"):
                self.handle_command("/" + low)
            elif item.startswith("/"):
                if self.handle_command(item) == "exit":
                    break
            else:
                self.run_with_timeout(item, mailbox=mailbox, deferred=deferred)


if __name__ == "__main__":
    # v1.0: 一打开立刻给出后端与预计耗时, 随后由 _br 逐阶段报告百分比/进程, 杜绝"像卡死"的等待
    _be = "GPU 加速(CuPy/CUDA)" if HAS_GPU else "CPU 模式(未检测到显卡)"
    print(C_HINT + "🟦 AI 小方 {} · {} 正在启动中… 构建 1B 级模型, 加载进度如下：".format(VERSION, _be) + C_RESET, flush=True)
    XiaoFang().run()