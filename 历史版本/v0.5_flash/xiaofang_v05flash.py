# -*- coding: utf-8 -*-
import sys, os, subprocess, datetime, random, re, time
import numpy as np
# ===== Flash GPU 后端: 能识别 RTX 就交给 CuPy, 否则回退 numpy =====
if not os.environ.get("XF_FORCE_CPU", ""):
    try:
        import cupy as _cp
        _cp.cuda.Device(0).use()
        XP = _cp
        HAS_GPU = True
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
import xiaofang_data_v05flash as DATA


VERSION = "0.5 Flash"
ENGINE_NAME = "Flphalit"
MODEL_NAME = "AI 小方 Flphalit 0.5 Flash"
STUDIO_NAME = "小方工作室"
STUDIO_SITE = "fanggame.company"
SEARCH_BACKENDS = ["duckduckgo", "bing", "brave", "google", "auto"]  # 多后端联网

# v0.5 正式版: 模型规模常量 (4096 维, 24 层, 32 头 = 4096/32 整除)
MODEL_D = 1024
MODEL_LAYERS = 12
MODEL_HEADS = 16
# 输入读取量: 即便长句也读更多词元, 避免"只读12字限制意图"
SEED_TOKENS = 24
DISPLAY_TOKENS = SEED_TOKENS
THINK_LINE_DELAY = 0.035   # 深度思考段逐行打字间隔(秒)


def _install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package,
                           "--quiet", "--break-system-packages"])


def _ensure_dependencies():
    required = {"colorama": "colorama", "ddgs": "ddgs",
                "pyfiglet": "pyfiglet", "numpy": "numpy"}
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


BLOCK_ICON = [
    "┌──────┐ ",
    "│   丨 丨    │   ",
    "│               │ ",
    "└──────┘ ",
]

C_BORDER = Fore.BLUE
C_THINK = Fore.LIGHTBLACK_EX
C_DEEP = Fore.LIGHTBLACK_EX   # v0.4.1 深度思考改为灰色
C_REPLY = Fore.LIGHTBLUE_EX   # v0.4.1 小方正式回答统一用浅蓝色
C_SYSTEM = Fore.YELLOW
C_ERROR = Fore.RED
C_HINT = Fore.LIGHTBLACK_EX
C_RESET = Style.RESET_ALL


def _typewrite(text, color=C_REPLY, delay=0.035, end="\n"):
    # 打字机效果: 逐字输出, 营造"一点点打出来"的感觉
    sys.stdout.write(color)
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(C_RESET + end)
    sys.stdout.flush()


def _typewrite_lines(lines, color=C_DEEP, line_delay=THINK_LINE_DELAY):
    # v0.5 Alpha: 深度思考段逐行打字输出 (长块用逐行更合适, 快而仍有点到点的感觉)
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


def show_startup():
    # v0.4Search: 简洁克制的 Claude Code 风格细框 + AI XIAOFANG ASCII 标题
    try:
        title = pyfiglet.figlet_format("AI XIAOFANG", font="slant")
    except Exception:
        title = pyfiglet.figlet_format("AI XIAOFANG", font="standard")
    title_rows = [l.rstrip("\n") for l in (title or "").rstrip("\n").split("\n")]
    title_rows = [l for l in title_rows if l.strip()] or ["AI XIAOFANG"]
    version_line = "  {} {}  · Deep Thinking · {}L×{}H d_model={} · Web Search · Multilingual  ".format(
        ENGINE_NAME, VERSION, MODEL_LAYERS, MODEL_HEADS, MODEL_D)
    content = [""]
    content.extend(title_rows)
    content.append("")
    content.append(version_line)
    content.append("")
    content.extend(BLOCK_ICON)
    content.append("")
    max_w = max(_disp_width(line) for line in content)
    inner = max_w + 6
    b = "─" * (inner - 2)
    framed = ["┌" + b + "┐"]
    for line in content:
        if line.strip():
            framed.append("│ " + line.ljust(inner - 4) + " │")
        else:
            framed.append("│ " + " " * (inner - 4) + " │")
    framed.append("└" + b + "┘")
    for line in framed:
        print(C_SYSTEM + line + C_RESET)
    print()
    print(C_HINT + "  ⚡ 输入 /help 查看命令   |   直接对话开始   |   深度思考中 ...  ⚡" + C_RESET)
    print()


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
            "has_tired": has_tired, "has_joy": has_joy, "has_love": has_love}

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
                ["模型", "AI", "ai", "机器人", "助手", "程序", "系统", "made"]):
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
        suggest_score = sum(1 for w in ["建议", "推荐", "怎么办", "该不该", "要不要", "帮我选", "怎么选"] if w in raw)
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

        # v0.5 正式版: 代码/算法意图
        code_score = 2 * sum(1 for w in ["写代码", "写个代码", "写程序", "写一段代码", "实现代码", "编程", "实现一个程序"]
                             if w in raw)
        code_score += 2 * sum(1 for w in ["排序", "冒泡", "快排", "快速排序", "二分查找", "二分", "递归", "阶乘",
                                          "斐波那契", "素数", "质数", "最大公约数", "欧几里得", "进制转换", "温度转换",
                                          "数据结构", "链表", "栈", "队列", "算法"] if w in raw)
        if re.search(r"帮我写.{0,6}(代码|函数|程序|排序|算法)", raw):
            code_score += 4

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


def _sinusoid(seq, d_model):
    pe = XP.zeros((seq, d_model), dtype=XP.float32)
    pos = XP.arange(seq)[:, None]
    i = XP.arange(d_model // 2)
    div = XP.power(10000.0, (2 * i) / d_model)
    pe[:, 0::2] = XP.sin(pos / div)
    pe[:, 1::2] = XP.cos(pos / div)
    return pe


class _F32RNG:
    """给 numpy 参数初始化外层套 float32: 4096 维 × 24 层的权重仅 f64 时超 40GB,
    转 f32 后压到可运行区间, 仍是纯 numpy、维度/层/头均不变."""
    def __init__(self, gen):
        self.gen = gen

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.gen.normal(loc, scale, size).astype(XP.float32)


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
        self.Wq = rng.normal(0, 0.02, (d_model, d_model))
        self.Wk = rng.normal(0, 0.02, (d_model, d_model))
        self.Wv = rng.normal(0, 0.02, (d_model, d_model))
        self.Wo = rng.normal(0, 0.02, (d_model, d_model))

    def forward(self, x):
        seq = x.shape[0]
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        Qh = Q.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        Kh = K.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        Vh = V.reshape(seq, self.n_heads, self.d_k).transpose(1, 0, 2)
        scores = Qh @ Kh.transpose(0, 2, 1) / XP.sqrt(self.d_k)
        mask = XP.triu(XP.full((seq, seq), -1e9), k=1)
        scores = scores + mask
        attn = _softmax(scores, -1)
        ctx = attn @ Vh
        ctx = ctx.transpose(1, 0, 2).reshape(seq, self.d_model)
        return ctx @ self.Wo, attn


class TransformerBlock:
    def __init__(self, d_model, d_ff, n_heads, rng):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rng)
        self.norm2 = LayerNorm(d_model)
        self.W1 = rng.normal(0, 0.02, (d_model, d_ff))
        self.b1 = XP.zeros(d_ff, dtype=XP.float32)
        self.W2 = rng.normal(0, 0.02, (d_ff, d_model))
        self.b2 = XP.zeros(d_model, dtype=XP.float32)

    def forward(self, x):
        a, attn = self.attn.forward(self.norm1.forward(x))
        x = x + a
        h = self.norm2.forward(x)
        h = _gelu(h @ self.W1 + self.b1)
        f = h @ self.W2 + self.b2
        x = x + f
        return x, attn


class DeepThinkTransformer:
    def __init__(self, vocab_list, d_model=MODEL_D, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS,
                 ngram_lm=None, seed=42):
        self.vocab = list(vocab_list)
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ngram = ngram_lm
        rng = _F32RNG(np.random.default_rng(seed))
        # 更深的权重算法: 更大的嵌入 + 更深的多层残差网络 (d_model=4096)
        self.embed = rng.normal(0, 0.02, (self.vocab_size, d_model))
        # FFN 中间倍数为 2, 配合 float32 把 24 层×4096 维权重压进可运行内存
        self.blocks = [TransformerBlock(d_model, d_model * 2, n_heads, rng)
                       for _ in range(n_layers)]
        self._ffn_norm = rng.normal(0, 0.02, (d_model,))
        self.W_out = rng.normal(0, 0.02, (d_model, self.vocab_size))
        self.b_out = XP.zeros(self.vocab_size, dtype=XP.float32)
        if ngram_lm and ngram_lm.uni_total > 0:
            for tok, count in ngram_lm.uni.items():
                idx = self.token2id.get(tok)
                if idx is not None:
                    self.b_out[idx] = np.log(count / ngram_lm.uni_total + 1e-8) * 1.6

        if HAS_GPU:
            self.embed = _to_dev(self.embed); self._ffn_norm = _to_dev(self._ffn_norm)
            self.W_out = _to_dev(self.W_out); self.b_out = _to_dev(self.b_out)
            for blk in self.blocks:
                blk.norm1.gamma=_to_dev(blk.norm1.gamma); blk.norm1.beta=_to_dev(blk.norm1.beta)
                blk.attn.Wq=_to_dev(blk.attn.Wq); blk.attn.Wk=_to_dev(blk.attn.Wk)
                blk.attn.Wv=_to_dev(blk.attn.Wv); blk.attn.Wo=_to_dev(blk.attn.Wo)
                blk.norm2.gamma=_to_dev(blk.norm2.gamma); blk.norm2.beta=_to_dev(blk.norm2.beta)
                blk.W1=_to_dev(blk.W1); blk.b1=_to_dev(blk.b1); blk.W2=_to_dev(blk.W2); blk.b2=_to_dev(blk.b2)
    def forward(self, token_ids, trace=False):
        if len(token_ids) == 0:
            token_ids = [0]
        seq = len(token_ids)
        x = self.embed[token_ids] + _sinusoid(seq, self.d_model)
        blocks = []
        attn_last = None
        for i, blk in enumerate(self.blocks):
            x, attn = blk.forward(x)
            attn_last = attn
            if trace:
                row = attn[0, -1, :]
                peaks = [float(attn[h, -1, :].max()) for h in range(self.n_heads)]
                blocks.append({
                    "norm": float(XP.linalg.norm(x[-1])),
                    "row": [round(float(v), 3) for v in row],
                    "peaks": [round(float(p), 3) for p in peaks]})
        # 深度权重细化: 学习到的逐元素深度门控, 对深层表征做 1+ε 尺度调制
        final = x[-1] * (1.0 + 0.02 * self._ffn_norm)
        logits = final @ self.W_out + self.b_out
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

    def score_distribution(self, tokens, bias_tokens=None):
        ids = [self.token2id.get(t, 0) for t in tokens]
        if not ids:
            ids = [0]
        probs, _ = self.forward(ids, trace=False)
        result = {}
        if self.ngram:
            t1 = tokens[-2] if len(tokens) >= 2 else NgramLM.START
            t2 = tokens[-1] if tokens else NgramLM.START
            ng = self.ngram.predict(t1, t2, bias_tokens)
            for tok in self.vocab:
                idx = self.token2id[tok]
                p = float(probs[idx])
                w = ng.get(tok, 0.0)
                result[tok] = p * (0.3 + w)
            if bias_tokens:
                for tok in bias_tokens:
                    if tok in self.token2id:
                        result[tok] = result.get(tok, 0.0) + 0.2
        else:
            for tok in self.vocab:
                result[tok] = float(probs[self.token2id[tok]])
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
        return n

    def count_output(self, tokens):
        n = len(tokens)
        self.output_tokens += n
        self.total += n
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
            inter = q & doc["tokens"]
            union = q | doc["tokens"]
            jaccard = len(inter) / len(union) if union else 0.0
            coverage = len(inter) / max(len(q), 1)
            bonus = 0.0
            if doc["title"] in query:
                bonus += 1.2
            for a in doc["aliases"]:
                if a in query:
                    bonus += 0.8
            score = jaccard + bonus + coverage * 0.3
            if score > min_score:
                scored.append((round(score, 3), doc["entry"]))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


_SENT_SPLIT = re.compile(r"[。；！？!?;\n]+")


def _segments_from_tokens(tokens):
    return "".join(tokens)


def _split_sents(text):
    parts = re.split(r"[。；！？!?]+", text)
    return [p.strip() for p in parts if p.strip()]


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
        strong_kb = bool(kb_hits and kb_hits[0][0] >= 0.6)
        if strong_kb:
            body = self._gen_from_kb(kb_hits, emo)
        elif emo["score"] <= -3 or intent["top"] in ("help", "study", "start", "suggest"):
            body = self._gen_comfort(emo, history)
        else:
            body = self._gen_freeform(query, emo, history)
        body = self._insert_emojis(body, emo)
        closer = self._pick_closer(emo)
        return " ".join(p for p in [opener, body, closer] if p)

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
        hit = self._kb_any_sentence(query)
        if hit:
            return hit
        return ("这个话题我本地还没存现成的讲法。你可以说『搜一下 {}』，"
                "我用联网把最新资料找出来，再给你整理成清楚的回答。").format(query[:10])

    def _kb_any_sentence(self, query):
        qset = self.tok.tokenize_set(query)
        if not qset:
            return None
        best = None
        for entry in DATA.KNOWLEDGE_BASE:
            for s in _split_sents(str(entry["b"])):
                st = self.tok.tokenize_set(s)
                if not st:
                    continue
                inter = len(qset & st)
                cov = inter / max(len(qset), 1)
                if cov >= 0.55 and (best is None or cov > best[0]):
                    best = (cov, s)
        if best:
            return best[1]
        return None

    def _insert_emojis(self, text, emo):
        segs = _segment_text(text, chunk_size=12)
        if not segs:
            return text
        prob = min(0.6, emo["intensity"] / 10.0 + 0.05)
        out = []
        for seg in segs:
            out.append(seg)
            if random.random() < prob:
                emoji = self._pick_emoji(emo)
                if emoji:
                    out.append(" " + emoji)
        return "".join(out)

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
        return False

    def _normalize(self, raw):
        s = raw.strip()
        for k, v in self.REP.items():
            s = s.replace(k, v)
        # 中文数字运算符 -> 英文运算符
        s = re.sub(r"乘以", "*", s); s = re.sub(r"除以", "/", s)
        s = re.sub(r"的平方", "**2", s); s = re.sub(r"的立方", "**3", s)
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

    def answer(self, raw):
        if not self.detect(raw):
            return None
        norm = self._normalize(raw)
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
    ]
    _GENERIC = ("def function_name(arr):\n    # 在这里写你的逻辑\n    result = []\n    for item in arr:\n        # 处理每一项\n        pass\n    return result",
                "思路：定义一个函数，接收输入，用循环/条件处理数据，最后返回结果。告诉我具体要做什么，我给你更完整的实现。")

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
            self.learned_words += len(add)
        defs = self._extract_definition(text, fresh_cands=cands)
        if defs:
            term, meaning = defs
            self._store(term, meaning)
            self.learned_defs += 1
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


class XiaoFang:
    def __init__(self):
        kb_words = []
        for e in DATA.KNOWLEDGE_BASE:
            kb_words.append(e["t"])
            kb_words.extend(e.get("a", []))
        # v0.5 Alpha 2: 人格词库单独读入分词器(不进 n-gram 正文, 避免泄露到中性问答)
        for e in getattr(DATA, "PERSONA_KB", []):
            kb_words.extend(e["kws"])
            kb_words.extend(e.get("r", []))
        self.tokenizer = Tokenizer(extra_words=kb_words)
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
        vocab_list = sorted(self.lm.vocab)
        self.transformer = DeepThinkTransformer(
            vocab_list, d_model=MODEL_D, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS, ngram_lm=self.lm)
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
        self.history = []
        self.max_history = 50
        # v0.4 正式版: 预设应答 (精确/包含关键词快速命中)
        self.presets = []
        for kws, reply in getattr(DATA, "PRESETS", []):
            self.presets.append({"kws": [k for k in kws if k], "reply": reply})

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

    def _strategy_scores(self, emo, intent, kb_hits):
        kb_score = kb_hits[0][0] if kb_hits else 0.0
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

        kb_top = kb_hits[0][1]["t"] if kb_hits else None
        kb_score = kb_hits[0][0] if kb_hits else 0.0

        seed = self.tokenizer.tokenize(user_input)[:SEED_TOKENS]   # v0.5: 读更多词元
        if not seed:
            seed = ["好"]
        bias = set(self.tokenizer.tokenize(user_input))
        if kb_hits:
            bias |= set(self.tokenizer.tokenize(kb_hits[0][1]["b"]))

        # ---------- 矩阵正演跟踪 ----------
        ids = [self.transformer.token2id.get(t, 0) for t in seed] or [0]
        _, tr = self.transformer.forward(ids, trace=True)

        top_idx = np.argsort(tr["probs"])[::-1][:5]
        top5 = [(self.transformer.vocab[i], float(tr["probs"][i])) for i in top_idx]

        dist = self.transformer.score_distribution(seed, bias)
        best_tok = max(dist, key=lambda k: dist.get(k, 0.0)) if dist else None
        best_prob = dist.get(best_tok, 0.0) if best_tok else 0.0

        detail = self._decompose(user_input, emo, intent, kb_hits, kb_top)
        strategies, kb_score = self._strategy_scores(emo, intent, kb_hits)
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
        # v0.5 Alpha: 深度思考段用逐行打字机打印; 打印结束后才计耗时, 使"已耗时"包含打字时长
        _typewrite_lines(parts, C_DEEP)
        elapsed = time.time() - t0
        tail = ["  ⏱ 本次思考已耗时 {:.2f} 秒 (含深度思考打字显示)".format(elapsed), "</deep_think>"]
        _typewrite_lines(tail, C_DEEP)
        return "\n".join(parts + tail)

    def _should_web(self, text, emo, intent, kb_hits):
        # v0.4Search: 自主判断是否联网 (联网过程内置于深度思考)
        kb_score = kb_hits[0][0] if kb_hits else 0.0
        if intent["top"] in ("search", "weather"):
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
        # v0.4Search: 已修复联网(改用新版 ddgs). 多后端顺序尝试 + 逐条卫护
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
        return answer + opener + learn_extra

    def reply(self, user_input, emo, intent, kb_hits):
        text = user_input.strip()
        # v0.5 Alpha: 官网/网站类问题只答官网 (不答模型身份)
        site_ask = any(m in text for m in ["官网", "网址", "fanggame", "fanggame.company"])
        if not site_ask and "网站" in text and any(m in text for m in
                ["小方", "工作室", "fanggame", "网站是多少", "官网是多少"]):
            site_ask = True
        if site_ask:
            return "小方工作室官网：{} 🎯 有新版本和更新都在那里～".format(STUDIO_SITE)
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
            if any(m in text for m in ["模型", "作者", "开发者", "谁做", "谁开发", "made"]):
                mline = random.choice([
                    "我是 {}，由 {} 研发的自主本地大模型引擎。💡".format(MODEL_NAME, STUDIO_NAME),
                    "我的模型名是 {}，归 {} 所有，是纯本地、保留版权的原创引擎。🔒".format(MODEL_NAME, STUDIO_NAME),
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

    def handle_command(self, cmd):
        c = cmd.strip().lower()
        if c == "/help":
            print(C_REPLY + "📖✨ 小方命令手册：" + C_RESET)
            print(C_REPLY + "  /help   - 显示所有命令" + C_RESET)
            print(C_REPLY + "  /clear  - 清空对话历史" + C_RESET)
            print(C_REPLY + "  /undo   - 撤回上一条对话" + C_RESET)
            print(C_REPLY + "  /exit   - 退出程序" + C_RESET)
            print(C_REPLY + "💡 引擎: DeepThink 深度思考 + {} {} 多层自注意力 Transformer(Self-Attn+LayerNorm+GELU+Softmax)+n-gram+语义检索".format(
                ENGINE_NAME, VERSION) + C_RESET)
            print(C_REPLY + "💡 {}层×{}头·d_model={}·词库10000+·含多语言 | 深度权重门控 | 预设应答 | emoji库 | 多后端联网 | Token计量 | 50条上下文 | 矩阵运算可视化".format(
                self.transformer.n_layers, self.transformer.n_heads, self.transformer.d_model) + C_RESET)
            return "ok"
        elif c == "/clear":
            self.history.clear()
            self.meter.reset()
            print(C_REPLY + "✨🧹 上下文+Token计量已清空，重新开始！" + C_RESET)
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
        elif c == "/exit":
            print(C_REPLY + "👋🌙 再见，{} {} 已退出。".format(ENGINE_NAME, VERSION) + C_RESET)
            return "exit"
        else:
            print(C_ERROR + "❓ 未知命令: " + cmd + "，输入 /help 查看。" + C_RESET)
            return "ok"

    def process_chat(self, user_input):
        text = (user_input or "").strip()
        # v0.5 正式版: 自升级大脑——从用户这句里学生词/学的解释
        learn_note = ""
        if text:
            lw, ld = self.learner.learn(text)
            if lw or ld:
                learn_note = "🧠 我记了 {} 个生词、{} 条知识。".format(lw, ld)
        # v0.4 正式版: 预设快速应答
        for p in self.presets:
            if any(k in text for k in p["kws"]):
                answer = p["reply"]
                out_tokens = self.tokenizer.tokenize(answer)
                self.meter.count_output(out_tokens)
                _typewrite("小方: " + answer)
                if learn_note:
                    _typewrite(C_HINT + learn_note + C_RESET)
                print(C_HINT + "  [{}]".format(self.meter.format()) + C_RESET)
                self.history.append(text)
                self.history.append(answer)
                return
        emo = self.emotion.analyze(text)
        intent = self.intent.detect(text, emo)
        kb_hits = self.retriever.retrieve(text)
        self.think(text, emo, intent, kb_hits)
        answer = self.reply(text, emo, intent, kb_hits)
        out_tokens = self.tokenizer.tokenize(answer)
        self.meter.count_output(out_tokens)
        if learn_note:
            answer = answer + " " + learn_note
        _typewrite("小方: " + answer)
        print(C_HINT + "  [{}]".format(self.meter.format()) + C_RESET)
        self.history.append(text)
        self.history.append(answer)
        self.lm.add_text(text)
        self.lm.add_text(answer)
        if len(self.history) > self.max_history:
            self.history = self.history[-(self.max_history):]

    def run(self):
        show_startup()
        while True:
            try:
                user_input = input(C_REPLY + "你: " + C_RESET).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                print(C_REPLY + "👋 小方再见～" + C_RESET)
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                if self.handle_command(user_input) == "exit":
                    break
            else:
                self.process_chat(user_input)


if __name__ == "__main__":
    XiaoFang().run()