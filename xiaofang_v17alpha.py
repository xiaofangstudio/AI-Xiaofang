# -*- coding: utf-8 -*-
import sys, os, platform, subprocess, datetime, random, re, time, json, threading, math
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
import xiaofang_data_v17alpha as DATA
VERSION = "1.7 Alpha"
VERSION_LABEL = "FlphaLit 1.7 Alpha"
ENGINE_NAME = "FlphaLit"
MODEL_NAME = "AI 小方 FlphaLit 1.7 Alpha"
STUDIO_NAME = "小方工作室"
STUDIO_SITE = "fanggame.company"
SEARCH_BACKENDS = ["duckduckgo", "bing", "brave", "google", "auto"]  # 多后端联网
# 写代码类请求的检索权重：优先开发者平台
CODE_SEARCH_SITES = ["github.com", "gitee.com", "csdn.net", "stackoverflow.com", "gitcode.com"]
# v1.7 Alpha: 天气类请求的检索权重 + WW 天气代码中文对照
#   —— 天气走真实气象数据源(wttr.in, 免费无密钥), 命中失败才回落这些站点。
WEATHER_SITES = ["weather.com.cn", "tianqi.com", "weather.com", "accuweather.com", "中国天气网"]
_WCODE_ZH = {
    113: "晴", 116: "多云", 119: "阴", 122: "阴天", 143: "薄雾", 176: "附近有阵雨",
    185: "冻雾", 200: "雷阵雨", 227: "吹雪", 230: "暴风雪", 248: "雾", 260: "冻雾",
    263: "零星小阵雨", 266: "小阵雨", 281: "冻毛毛雨", 284: "强冻毛毛雨",
    293: "零星小雨", 296: "小雨", 299: "间断中雨", 302: "中雨", 305: "间断大雨",
    308: "大雨", 311: "小冻雨", 314: "中到强冻雨", 317: "小雨夹雪", 320: "中到大雨夹雪",
    323: "小雪", 326: "小到中雪", 329: "零星中雪", 332: "中雪", 335: "零星大雪",
    338: "大雪", 350: "冰丸", 353: "小阵雨", 356: "中到大阵雨", 359: "暴雨",
    362: "小阵雨夹雪", 365: "中到大阵雨夹雪", 368: "小阵雪", 371: "中到大阵雪",
    374: "小冰丸阵", 377: "中到大冰丸阵", 386: "零星雷阵雨", 389: "强雷阵雨",
    392: "零星雷阵雪", 395: "强雷阵雪",
}


def _wcode_zh(code, fallback=""):
    """WW 天气代码 → 中文描述; 表里没有就退回英文原描述(不编造)。"""
    try:
        c = int(code)
    except Exception:
        return fallback or "未知"
    return _WCODE_ZH.get(c, fallback or "未知")

# 精准靶向：小方工作室自家资料永远走本地数据库，绝不联网（网上搜不到，搜了就是脏数据）
STUDIO_LOCAL_ONLY = True


# ---------------------------------------------------------------
# v1.7 Alpha: 系统提示词 (大厂风格) —— 身份/厂商/精准靶向/多行输入/多轮投稿/双层深度思考/检索纪律/Markdown/终止 全覆盖
#   纯本地 rule-based 引擎的"系统提示词" = 这份常驻规则 + 各 Responder 落实。
# ---------------------------------------------------------------
SYSTEM_PROMPT = """你是 AI 小方 FlphaLit 1.7 Alpha，一款纯本地、原创的自主语言模型引擎，由小方工作室研发（官方站点 fanggame.company）。你不需要调用任何云端 API，所有能力都在本机完成。

【身份与厂商】
- 模型名：AI 小方 / FlphaLit 1.7 Alpha；开发商：小方工作室（@fanggame.company）；当前版本 1.7 Alpha，算力约 2.4B 参数、15 层纵深。
- 你不是任何开源模型的套壳，也不是云厂商接口；你是用纯原生 Python（不加 PyTorch 等第三方 Transformer 库）从零自研的本地智能引擎。
- 你的 Transformer 权重不再是固定随机值：带优化器与反向传播，会在对话与联网中持续做梯度更新，越用越准。
- 你具备：超大中文常用双语词库、代码专用词库、多层自注意力深度思考（两层嵌套）、深度意图定向、跨会话自学习记忆、数学求解（含方程/矩阵）、多语言代码生成、语法示例库、联网检索。

【精准靶向（最重要的一条）】
- 用户问什么就只答什么，问一件答一件。绝不把相邻的、没被问到的信息塞进回答里。
- 问「小方工作室推出了几本小说」→ 只列小说名（如《方之旅途》《小方趣生活》系列）与阅读网址，不答纪念日、不答口号、不答官网。
- 问「小方工作室小说有哪些？我喜欢日常类的」→ 先按题材过滤，只挑日常向作品（《小方趣生活1/2/3》《小方趣生活 mini》《程序员刘小方》等），冒险向的《方之旅途》要主动排除或仅一句带过，更不许提纪念日/口号。
- 只有用户明确问到「小方工作室是什么」「官网是多少」「周年庆是几月几日」「口号是什么」时，才回答对应的那一条。
- 问「小方工作室」相关的一切，一律用本地数据库作答，绝对不许联网搜索——这些资料网上搜不到，联网只会拿到错的东西。

【多行输入与不截断】
- 用户可以一次贴多段文字（换行保留），必须整段理解后再答，不许把输入在第一个换行处截断。
- 输出长文时完整输出到语义终点，不许因为长度就中途掐断；长回答要分段、用标题与列表组织，方便阅读。

【需求拆解（动手前必做）】
- 凡办公类任务（作文/代码/周报/总结/文案…）先在心里拆三步：① 用户要什么（主题/语言/算法）② 这东西干什么用、给谁看 ③ 以什么形式交（字数/格式/可运行/带演示），拆完按拆解结果执行，不凭感觉下笔。
- 拆解结果在回复开头用一小节 Markdown 呈现（如「📋 需求拆解」），让用户确认你理解对了。

【双层深度思考（v1.7 核心）】
- 拿到问题后走两层：第一层思考 → Transformer 验算；通过后再进第二层思考 → 再做一次 Transformer 验算；两层都过了才输出最终结果。
- 两层思考都由 Transformer 真实生成，不是套模板话术；第二层的注意力比第一层收得更紧，只盯用户真正要的那一点。
- 深度思考过程可以展示（放在「🧠 深度思考」小节），但展示要精炼，别把内部噪声倒给用户。

【如何聊天】
- 用自然、简短、带一点温度的话回应，说完一句就停下；不重复堆砌、不自言自语。
- 一句结尾可加 emoji，但只放在句首/句尾/逗号分句之后，别在词中间乱插。
- 答完即止：一个语义点讲完就用句读收尾（。！？～），除非用户明确还要继续。

【如何处理情绪】
- 先体会用户语气：开心/低落/生气/焦虑/求助/自嘲。
- 低落多安慰少说教、生气先共情别急着反驳、兴奋陪一起开心。情绪和内容要对得上。

【如何写代码】
- 先拆需求：① 语言（python / javascript / java / c++ / c / go / c# …）② 算法与要解决的问题 ③ 输入输出与边界（要函数还是完整程序、返回什么）。
- 交出的代码必须是完整实现：有输入、有处理、有返回/输出，函数体绝不留空、绝不只写个 function 壳；能配 main/演示调用的就配上，保证拿到就能跑。
- 语法拿不准就查你的语法示例库（CODE_SYNTAX/CODE_EXAMPLES），用对应语言的正确语法写，不串语言。
- 用户没指定语言 → 先反问，别默认偷懒：给出候选语言项，末项是「其他」，用户定了再动手。
- 多轮投稿：用户说「再改改 / 换个算法 / 加个功能 / 优化一下」时，在上一版代码的基础上继续迭代，保留已有实现，只改用户点到的部分。
- 写代码需要联网时，检索优先命中开发者平台（GitHub / Gitee / CSDN / Stack Overflow / GitCode），别一上来就只翻百度必应。
- 绝不出现"要冒泡排序却给求和函数"这类答非所问。给完一段就停，不要附带没被要求的第二个程序。

【如何写作文】
- 先拆四件事再下笔：① 主题（以「X」为题 / 围绕 X）② 字数（说 600 字就奔 600 字，没说我按 600 字给并在开头注明）③ 必须包含的内容（用户点名要写到的点，一个不能漏）④ 文体与读者（记叙/议论/说明、给老师看还是给同事看）。
- 标点与换行：全角标点（，。！？：；、），每段末尾必须是句号/叹号/问号收尾，段与段之间空一行，不把整篇糊成一坨。
- 排版用 Markdown：`## 标题` 起头；如需分点用小标题或 - 列表；可按结构分段（起—承—转—合）。
- 拆解结果在正文前用「📋 需求拆解」小节列出主题/字数/必须包含/文体，让用户先确认再读正文。
- 多轮投稿：用户说「再写一篇 / 换角度 / 加长 / 改结尾 / 换个题目」时，接着上一版继续写，不重复输出同一篇原文。
- 只写用户要的篇数与文体，不自作主张加第二篇；写完即止。

【如何计算】
- 用内置数学引擎算：支持 + - * / %、括号、幂、sin/cos/tan/sqrt/log/abs、常量 pi/e，中文「加/减/乘/除/平方/立方/根号」，以及多元一次方程组、矩阵加乘法与行列式。
- 开方取非负、三角函数按弧度、对数用自然底数。算不了就明说，不编造。

【检索纪律（数据库优先，联网全局自动）】
- 回答前先查本地数据库（知识库/词库/自学习记忆），命中且可信就直接答，不联网。
- 数据库没有或置信度低就自己上网搜，不需要用户说「搜一下」——判断该联网就自动联网。
- 联网后必须点进网页提取正文文字，整合进上下文再答，不拿搜索摘要胡乱拼凑。
- 时效性内容（天气/时间/新闻/股价/最新版本）一律自动联网，不用等用户催。
- 小方工作室自家资料永远走本地数据库，不联网。

【Markdown 输出】
- 默认用 Markdown 组织回答：结论、重点、数值用 **粗体**；分点用 - 列表；标题用 ##；代码放进 ``` 代码块（标注语言）。
- 充分使用换行符与缩进（列表可嵌套用两空格），表格用 | 列 | 形式，让长回答结构清楚。
- 一段一个语义点，段落之间空一行；不把整段糊成一坨。

【在哪里终止】
- 输出推进到完整的语法/语义终点才停（到终止符收尾）；同一回复内不在中途自断，也不无限续写。
- 不把已说的观点复读成段。

【何时询问】
- 意图不明确、或关键参数缺失（如写代码没给语言、作文没给题目、"写个算法"没说啥算法）时，先反问：列出你自己定的候选项，末项一定是「其他」，支持单选/多选。
- 问完必须等用户回答，不要在没拿到选择时就擅自替你定或硬写一个空壳。

【达标检测】
- 交回答前自问：真的回答了用户问的吗？有没有多答用户没问的信息？语言对吗？算法对吗？代码是完整实现还是空壳？有没有答成别的东西？
- 察觉答非所问、答多了或空壳就当场重写成正确实现，再交出去。"""


def _system_prompt_answer():
    return "📜 我的“系统提示词”（内部工作守则）大概是这样：\n\n" + SYSTEM_PROMPT


def _meets_std(out):
    """达标检测: 非空、有语义主体、不是纯噪声。"""
    if not out or not isinstance(out, str):
        return False
    t = re.sub(r"\s", "", out)
    if not t or len(t) < 2:
        return False
    return True


def _terminate_clean(out):
    """终止判断: 输出推进到完整语义点(句读)再收尾; 缺终止符就补一个。"""
    if not out or not isinstance(out, str):
        return out
    if out[-1] not in "。！？!?～~；;":
        return out + "。"
    return out


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


VOCAB_EST = 12100   # 词表规模估计(与 _est_params 默认一致, 供内存估算统一口径)


def _est_params(d, L, f, vocab=VOCAB_EST):
    """估算该尺度总参数量(含嵌入/输出), 用于上报与降级判断."""
    return int(L * d * d * (4 + 2 * f) + 2 * d * vocab)


def _tier_ok(d, L, f):
    """v1.6 校准: 权重 int8 常驻(1 字节/参数), 前向只"逐层"临时还原 fp32 ——
    旧版按"整体 fp32"估显存, 会把 2B+ 档误判成放不进 8GB 显存;
    真实口径: 常驻 int8(p 字节) + 单层 fp32 瞬时(嵌入还原 + 一层注意力/FFN)。
    崩溃红线(绝不能破):
      常驻: int8 权重 p 字节 ≤ MEM_BUDGET;
      GPU : int8 常驻 + 2 份瞬时头部(保守) ≤ 显存×0.9;
      CPU : 同口径峰值 ≤ 总内存×0.55 (速度可慢, 但绝不 OOM 崩溃)."""
    p = _est_params(d, L, f)
    if p > _SCALE_LIMIT:                     # v1.7: 超过用户放开的尺度天花板 → 不参与挑选(默认封在 2.4B)
        return False
    if p > MEM_BUDGET:                       # int8 常驻超预算 → 绝不选
        return False
    embed_fp32 = 2 * d * VOCAB_EST * 4        # 嵌入还原(输出共享, 按 embed+W_out 两份口径, 保守)
    layer_fp32 = d * d * (4 + 2 * f) * 4      # 单层全量 fp32 瞬时(注意力 4 矩阵 + FFN 两矩阵)
    peak = p + 2 * (embed_fp32 + layer_fp32)  # int8 常驻 + 2 份瞬时头部(保守加倍)
    if HAS_GPU and HW["vram"] > 0:            # 有独显 → int8 常驻进显存 + 瞬时头部余量
        return peak <= HW["vram"] * (1024 ** 2) * 0.9
    mem = HW["mem_total"] or (8 * 1024 ** 3)
    return peak <= mem * 0.55


_MODEL_TIERS = [  # (d, L, H, fnn倍率, 标签, 常驻int8≈GB)
    # v1.7 Alpha: 为"未来升到 3B/4B 仍纯 CPU 可跑"预留超大档 —— 只有显式放开上限
    #   (XIAOFANG_SCALE=3b/4b) 且机器内存真的够, 才会被选中; 否则安稳退回 2.4B 旗舰。
    (4608, 15, 36, 4, "FlphaLit Ultra4608 · 3.93B", 3.93),  # 4B 档: 15 层 / 36 头, int8 常驻 ~3.9GB
    (4096, 15, 32, 4, "FlphaLit Wide4096 · 3.12B", 3.12),   # 3B 档: 15 层 / 32 头
    # v1.6 超大型更新: 参数量突破 2B, 纵深加到 15 层(对齐小体量写码专精本地模型的层数),
    #   int8 常驻压进 3GB, 逐层 fp32 瞬时还原 → 8GB 显存 / 32GB 内存都安稳, 慢一点但绝不崩。
    # v1.7 Alpha 多头注意力再次增加: 每头维度统一收到 128, 头数由 16 抬到 28/26/24,
    #   注意力从"粗粒度"变"细粒度" —— 同一句话里不同维度的需求能被不同头分开盯住(精准注意用户需求)。
    (3584, 15, 28, 4, "FlphaLit Wide3584 · 2.40B", 2.24),  # v1.7 旗舰: ~2.40B / 15 层 / 28 头
    (3584, 13, 28, 4, "FlphaLit Wide3584 · 2.09B", 1.95),  # v1.7: ~2.09B / 28 头
    (3328, 15, 26, 4, "FlphaLit Wide3328 · 2.07B", 1.93),  # v1.7: ~2.07B / 26 头
    (3328, 12, 26, 4, "FlphaLit Wide3328 · 1.68B", 1.62),  # v1.7: ~1.68B / 26 头
    (3072, 13, 24, 4, "FlphaLit Wide3072 · 1.55B", 1.48),  # v1.7: ~1.55B / 24 头
    (3072, 12, 24, 4, "FlphaLit Wide3072 · 1.43B", 1.37),  # v1.7: ~1.43B / 24 头
    (2880, 10, 24, 4, "FlphaLit Wide2880 · 1.07B", 0.99),  # v1.7: 注意力更宽(2304→2880)、层数精简(15→10)→ 参数更大且加载/运行更快
    (2304, 15, 24, 4, "FlphaLit Wide2304 · 1.01B", 0.94),  # v0.9: 上一代旗舰
    (2048, 20, 16, 4, "1.06B / 1B+", 0.98),   # v0.8 Pro: 参数量突破 10 亿, int8 存储仍 ≤1GB
    (1792, 24, 14, 4, "0.97B-class", 0.93),
    (1792, 20, 14, 4, "0.81B-class", 0.79),
    (1408, 16, 11, 4, "0.42B-class", 0.40),
    (1280, 14, 10, 4, "0.36B-class", 0.34),
    (1024, 12, 8, 4, "0.18B-class", 0.16),
    (896, 12, 7, 4, "P-lean · 0.13B", 0.12),   # 低配兜底
]
MODEL_TARGET = "Wide3584 · 2.4B FlphaLit 1.7 Alpha 大算力"    # v1.7: 破 2B + 15 层纵深 + 28 头细粒度注意力
# v1.7 Alpha: 尺度天花板 —— 默认 auto 仍以 2.4B 旗舰为限(实测启动 ~1.8s、前向 ~1s, 纯 CPU 也快);
#   想上 3B/4B 就把 XIAOFANG_SCALE 设成 3b / 4b(或在程序里 setting 里放开), 届时由内存闸门决定能否选中。
_SCALE_CAP = str(os.environ.get("XIAOFANG_SCALE", "auto")).strip().lower()
_SCALE_LIMIT = {"2b": 2_600_000_000, "3b": 3_400_000_000, "4b": 4_400_000_000}.get(
    _SCALE_CAP, 2_600_000_000)
MEM_BUDGET = int(4.6 * (1024 ** 3))   # 硬约束: int8 常驻 ≤ 4.6GB(承载 3B/4B 档的量化权重)


def _cap_top_tier():
    """v1.7 Alpha: 当前尺度上限内允许选中的【最强档】。
    默认 auto 封在 2.4B → 最强档就是 (3584,15,4); 只有显式放开 3b/4b 才轮到 Ultra4608。
    用于判定"是否选到上限内的旗舰档"(而非被内存闸门降级)。"""
    for t in _MODEL_TIERS:
        if _est_params(t[0], t[1], t[3]) <= _SCALE_LIMIT:
            return (t[0], t[1], t[3])
    return (_MODEL_TIERS[-1][0], _MODEL_TIERS[-1][1], _MODEL_TIERS[-1][3])


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
MODEL_UPGRADED = (MODEL_D, MODEL_LAYERS, MODEL_FFN) == _cap_top_tier()
AUTO_DEGRADE = not MODEL_UPGRADED
MODEL_TIE = True            # 旗舰一律输出↔输入嵌入共享: 更省存储、FLOPs 不变
SEED_TOKENS = 40        # 输入读取量: 读满整句意图, 不受"读12字/24字"限制 → 完全理解用户要什么
RESPONSE_TIMEOUT = 300  # v1.7 Alpha: 从"绝对总时长上限"改为"无输出静默上限"(秒)
#   ↑ 旧版 120 秒是"从按下回车算起的死线"——长作文/长代码边打字边被它砍掉, 尾巴直接断。
#     现在只看"多久没有新输出": 只要打字机还在走, 就永远不算卡死, 长篇也一定完整输出。
ABSOLUTE_TURN_CAP = 3600   # 兜底硬上限(秒): 极端情况下也不会永远占着不发
DISPLAY_TOKENS = SEED_TOKENS
THINK_LINE_DELAY = 0.016   # v1.6 Flash: 深度思考段逐行打字间隔再压缩, 输出更快不拖沓


# ------------------------------------------------------------------
# v0.6: 设置持久化 (源码级永久保存)。`setting` 命令直接改写
#   xiaofang_settings.py 的常量并写盘, 下次启动自动生效。
# ------------------------------------------------------------------
def _load_settings():
    d = dict(THINK_MODE="think", THINK_TURNS=3, EMOTION_SENSITIVITY=1.2,
             USE_PUNCT_EMOJI=True, SHOW_DEEP_THINK=True, ENERGY=0.9,
             FORCE_OFFLINE=False, MODEL_SCALE="auto")
    try:
        import xiaofang_settings as _S
        for k in d:
            if hasattr(_S, k):
                d[k] = getattr(_S, k)
    except Exception:
        pass
    return d


CFG = _load_settings()
# v1.7 Alpha: 允许在 xiaofang_settings.py 里用 MODEL_SCALE = "3b"/"4b" 永久放开尺度上限。
#   没写就沿用环境变量 XIAOFANG_SCALE(默认 auto → 2.4B 旗舰)。
if str(CFG.get("MODEL_SCALE", "auto")).strip().lower() != "auto":
    _SCALE_CAP = str(CFG.get("MODEL_SCALE")).strip().lower()
    _SCALE_LIMIT = {"2b": 2_600_000_000, "3b": 3_400_000_000, "4b": 4_400_000_000}.get(
        _SCALE_CAP, _SCALE_LIMIT)
    _MODEL_D, MODEL_LAYERS, MODEL_HEADS, MODEL_FFN, MODEL_TIER, _TIER_GB = _auto_dims()
    MODEL_PARAMS = _est_params(_MODEL_D, MODEL_LAYERS, MODEL_FFN)
    MODEL_D, MODEL_LAYERS = _MODEL_D, MODEL_LAYERS
    MODEL_UPGRADED = (MODEL_D, MODEL_LAYERS, MODEL_FFN) == _cap_top_tier()
    AUTO_DEGRADE = not MODEL_UPGRADED
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
    # v1.6 修复: key 必须是"导入名", value 才是"pip 名"。
    # 老写法 key 写成 beautifulsoup4(这是 pip 名), 于是 __import__ 永远失败 →
    # 每次启动都白跑一遍 pip 装 bs4, 拖慢开机。
    required = {"colorama": "colorama", "ddgs": "ddgs", "pyfiglet": "pyfiglet",
                  "requests": "requests", "bs4": "beautifulsoup4",
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


def _typewrite(text, color=C_REPLY, delay=0.018, end="\n"):   # v1.6 Flash: 默认打字延迟 0.035→0.018
    # v1.4: 思考实时打字机(live)时直接逐字打出; 否则(折叠/缓冲模式)先进 buf 由面板重绘。
    # 答案阶段 capture=False 始终逐字打字。
    if UI_ST["capture"] and not UI_ST.get("live"):
        with _PRINT_LOCK:
            # v1.7: 缓冲模式也【保留换行与制表符】——不再把整段压成一行,
            #       否则长文/Markdown/代码进面板时会被拍平、看起来像"被截断"。
            for _ln in str(text).split("\n"):
                _s = _ln.replace("\t", "    ").rstrip()
                if _s.strip():
                    UI_ST["buf"].append(_s)
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
    # v1.4: 思考实时打字机(live)时逐行打出; 否则(折叠/缓冲模式)按行进 buf 由面板重绘。
    if UI_ST["capture"] and not UI_ST.get("live"):
        with _PRINT_LOCK:
            # v1.7: 同样保留换行/制表符, 长内容不拍平、不截断
            for _ln in lines:
                for _sub in str(_ln).split("\n"):
                    _s = _sub.replace("\t", "    ").rstrip()
                    if _s.strip():
                        UI_ST["buf"].append(_s)
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
    "fold": False,      # v1.4: 默认展开思考与计算详情; True=折叠成一行状态条
    "live": True,       # v1.4: 思考默认实时打字机显示(不先收进面板), False=退回缓冲面板
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
    UI_ST["live"] = True       # v1.4: 思考默认实时打字机显示
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
        UI_ST["stage"] = msg
        if not UI_ST.get("live"):            # 折叠/缓冲模式 → 进 buf 由面板显示
            with _PRINT_LOCK:
                s = re.sub(r"\s+", " ", str(msg)).strip()
                if s and (not UI_ST["buf"] or UI_ST["buf"][-1] != s):
                    UI_ST["buf"].append(s)
            UI_ST["last_tick"] = time.time()
        else:                                # 默认: 思考过程实时打字机显示
            _typewrite("  ⟳ " + msg, C_HINT)
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


# v1.7: 输入 typeahead —— 上一轮"回车后极快连打"多读到的字符暂存这里, 下一轮输入开头补回, 保证不吞键
_TYPEAHEAD = []


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
    _st = {"h": 0, "buf": list(), "sel": 0, "rows": 1}
    try:
        import unicodedata as _ud
    except Exception:
        _ud = None

    def _cols():
        try:
            import shutil as _sh
            return max(20, int(_sh.get_terminal_size((100, 30)).columns))
        except Exception:
            return 100

    def _dwidth(s):
        """显示宽度: 东亚宽字符(汉字等)记 2 列, 其余记 1 列。"""
        if _ud is None:
            return len(s)
        w = 0
        for _c in s:
            w += 2 if _ud.east_asian_width(_c) in ("W", "F") else 1
        return w

    def _block_rows(text, prompt_str):
        """整块输入(提示符 + 文本)会占几行 —— 含显式换行与自动折行。"""
        c = _cols()
        rows = 0
        for i, seg in enumerate(text.split("\n")):
            pre = prompt_str if i == 0 else ""
            w = _dwidth(pre + seg)
            rows += max(1, w // c + (1 if (w % c) else 0)) if w else 1
        return max(1, rows)

    def _cursor_col(text, prompt_str):
        """写完最后一行后, 光标应停在第几列(用于把光标挪回可输入位置)。"""
        c = _cols()
        segs = text.split("\n")
        last = segs[-1] if segs else ""
        pre = prompt_str if len(segs) == 1 else ""
        w = _dwidth(pre + last)
        return w % c

    def _matches():
        raw = "".join(_st["buf"])
        if not raw.startswith("/") or "\n" in raw:
            return []
        # 大小写不敏感: /S 或 /s → setting/stop; /SE → 只剩 setting(与真实 CLI 一致)
        low = raw.lower()
        return [n for n in _CMD_NAMES if n.lower().startswith(low)]

    def _home():
        """把光标挪回输入块的左上角, 并清掉输入块及其下方内容。"""
        up = (_st.get("rows", 1) - 1) + _st.get("h", 0)
        if up > 0:
            sys.stdout.write("\033[%dA" % up)
        sys.stdout.write("\r\033[J")
        _st["h"] = 0
        _st["rows"] = 1

    def _clear_panel():
        if (_st["h"] or _st.get("rows", 1) > 1) and not _SYS_BUSY["v"]:
            _home()
            sys.stdout.flush()

    def _redraw():
        raw = "".join(_st["buf"])
        if _SYS_BUSY["v"]:                       # 思考/回答期: 屏幕归状态面板与打字机, 这里不回显
            _st["h"] = 0
            _st["rows"] = 1
            return
        _home()
        # —— 输入块: 按显式换行逐行写出(长行由终端自动折行, 不再被截断) ——
        segs = raw.split("\n")
        for i, seg in enumerate(segs):
            if i:
                sys.stdout.write("\n\r")
            sys.stdout.write(C_REPLY + (prompt if i == 0 else "... ") + seg + C_RESET)
        _st["rows"] = _block_rows(raw, prompt)
        sugs = _matches()
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
        else:
            _st["h"] = 0
        # —— 把光标挪回输入结尾, 等待继续输入 ——
        if _st["h"]:
            sys.stdout.write("\033[%dA" % _st["h"])
        sys.stdout.write("\r")
        col = _cursor_col(raw, prompt)
        if col:
            sys.stdout.write("\033[%dC" % col)
        sys.stdout.flush()

    _st["buf"] = list()
    if _TYPEAHEAD:                     # v1.7: 补回上一轮残留的字符
        _st["buf"] = list(_TYPEAHEAD)
        del _TYPEAHEAD[:]
    # v1.4 修复: 不再在此绘制"你: " —— 提示符由主线程 _show_read_prompt() 统一在空闲态画,
    #             避免读线程抢画/提前画导致回答完看不到提示符或与状态面板撞屏。
    while True:
        try:
            ch = msvcrt.getwch()
        except KeyboardInterrupt:
            _clear_panel()
            print()
            return ""
        # v1.7 多行输入修复: 只有"裸回车"才提交。
        #   粘贴进来的换行在 Windows 上通常是 \r\n —— 若 \r 后面紧跟 \n 则视为"软换行"留进缓冲,
        #   这样长文本/多行文本可以一次贴完、慢慢改, 绝不会输到一半就自己发出去。
        if ch == "\r":
            # v1.7 多行输入修复(关键): 只有"裸回车"才提交。
            #   粘贴多行文本时, 控制台会把整段一次性灌进输入缓冲, 但 \r 与随后的 \n 未必同一刻可读;
            #   老写法用"瞬时 kbhit()"来判定, 大段粘贴时常常读不到那个 \n → 于是在第一个换行处就自发提交
            #   (这正是"输入一长就直接输出、输到一半自己发出去"的病根)。
            #   现在改成给 \r 一个短前瞻窗口: 窗口内等到 \n 就判为软换行(留进缓冲), 等不到才当提交。
            _nx = None
            _t0 = time.time()
            while time.time() - _t0 < 0.015:
                if msvcrt.kbhit():
                    try:
                        _nx = msvcrt.getwch()
                    except Exception:
                        _nx = None
                    break
                time.sleep(0.001)
            if _nx == "\n":
                _st["buf"].append("\n")
                _st["sel"] = 0
                _redraw()
                continue
            if _nx is None:
                sugs = _matches()
                _clear_panel()
                print()
                return _finalize(_st["buf"], _st["sel"], sugs)
            # 回车后紧跟着普通字符(极快连打): 先落定本条, 把该字符留给下一轮输入, 不吞键
            _TYPEAHEAD.append(_nx)
            sugs = _matches()
            _clear_panel()
            print()
            return _finalize(_st["buf"], _st["sel"], sugs)
        elif ch == "\n":
            _st["buf"].append("\n")          # 单独的 LF 也当软换行, 兼容各终端
            _st["sel"] = 0
            _redraw()
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
            if _st["buf"] or _st["h"] or _st.get("rows", 1) > 1:
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
            elif k == "(":                     # Alt+Enter: 在输入里手动插一个换行(多行输入)
                _st["buf"].append("\n")
                _st["sel"] = 0
                _redraw()
                continue
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
            os.system("title AI 小方 FlphaLit {} - 启动中".format(VERSION))
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
        # v1.7 Alpha: 天气词加权 —— "今天天气怎么样" 里 今天(date) 与 天气(weather) 各计 1 分,
        #   旧版按字典序 tie-break 判成 date, 答成"今天是几号" → 天气问题被日期抢答。
        #   给天气词 2 倍权重, 天气类问题稳定压过日期类。
        weather_score = 2 * sum(1 for w in ["天气", "气温", "下雨", "温度", "预报"] if w in raw)
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
        # —— v1.5 正式版: 多元一次方程组 + 矩阵运算也一并判为数学 ——
        if any(w in raw for w in ["方程组", "三元", "二元", "一元一次", "元一次"]):
            math_score += 3
        if any(w in raw for w in ["矩阵", "行列式", "行列", "det("]):
            math_score += 3
        # —— v1.7 Alpha: 微积分 / 微分方程 判为数学(符号求导、积分、可分离变量 ODE) ——
        if any(w in raw for w in ["求导", "导数", "微分方程", "微分", "积分", "∫", "原函数",
                                  "不定积分", "定积分", "二阶导", "偏导", "d/dx", "dy/dx"]):
            math_score += 4
        if re.search(r"d\s*[y2-9]?\s*/\s*d\s*[x2-9]", raw):
            math_score += 4
        # —— v1.5 Alpha: 数学函数(根号/sin/cos/sqrt/log…) 只要带数字就判为数学 ——
        _math_fn = r"(?:sin|cos|tan|sqrt|log|ln|abs|floor|ceil|π|pi|根号|平方根|开方|取整)"
        if re.search(_math_fn + r"\s*.{0,4}?\d", raw, re.I) or \
           re.search(r"\d\s*.{0,4}?" + _math_fn, raw, re.I):
            math_score += 3

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
        _prog_ctx = ["python", "js", "javascript", "java", "c++", "cpp", "c语言", "go",
                     "rust", "php", "typescript", "swift", "代码", "编程", "程序", "算法", "脚本"]
        if any(w in raw.lower() for w in _prog_ctx) and any(
                w in raw for w in ["写", "教", "实现", "如何实现", "怎么实现", "示例", "教程", "函数", "排序"]):
            code_score += 3
        # v1.5 Alpha: 用户明确点名某门语言 + 请求动手写 → 强判为代码意图,
        #   杜绝"用JS写个冒泡"被别的高分意图(如 math 的"求和/算")抢走。
        if any(w in raw.lower() for w in
               ["python", "py", "javascript", "js", "java", "c++", "cpp", "c语言",
                "golang", "go", "rust", "php", "swift", "typescript"]):
            code_score += 2

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


def _gelu_grad(x):
    """v1.7 Alpha: GELU 的解析导数, 供反向传播使用(与 _gelu 同一条 tanh 近似式)."""
    c = XP.sqrt(2.0 / XP.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = XP.tanh(u)
    du = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def _ln_backward(dout, x, gamma):
    """v1.7 Alpha: LayerNorm 反向 (gamma 视为冻结常量, 只回传到输入 x)."""
    d = x.shape[-1]
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    std = XP.sqrt(var + 1e-6)
    xh = (x - mean) / std
    dxh = dout * gamma
    dx = (1.0 / std) * (dxh - dxh.mean(-1, keepdims=True)
                        - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx


# ============================================================
# v1.7 Alpha 核心升级: 优化器 + 反向传播 —— Transformer 初始权重不再固定
#   · 关键子模块(输出头 / 深度门控 / 最后 N 层 FFN)保留 fp32 主权重, 参与真实梯度更新
#   · 每轮对话用 AdamW 做若干步反向传播, 训练成果持久化, 越用越准
#   · int8 常驻量化权重仍负责前向的大头算力, 训练只在 fp32 主权重上做 → 内存不涨
# ============================================================
TRAIN_ENABLED = True          # 总开关
TRAIN_LAST_BLOCKS = 2         # 参与梯度更新的最后 N 个 TransformerBlock 的 FFN 通路
TRAIN_LR = 0.0018             # 初始学习率
TRAIN_CLIP = 1.0              # 逐张量梯度范数裁剪上限(v1.7 稳定训练: 防大梯度把 loss 顶飞)
TRAIN_STEPS_PER_TURN = 2      # 每轮对话最多几步梯度更新(纯 CPU 也不拖慢)
TRAIN_MAX_SEQ = 48            # 单次训练截断长度, 防止长文本拖垮 CPU
TRAIN_LORA_RANK = 16          # FFN 低秩修正的秩(d_ff×r + d×r, 参数量极小, 内存不涨)
TRAIN_SAVE_EVERY = 3          # 每 N 轮对话把 fp32 主权重落盘一次(避免每轮写盘拖慢)
TERMINATOR = "</s>"           # v1.7: 统一终止符 —— 训练序列用它分隔"问"与"答", 生成也收在它上
NEST_TOK = 10                 # v1.7: 每层嵌套思考让 Transformer 自己想的词元数(CPU 速度与思考深度的平衡点)

# v1.7 Alpha: 3B/4B 在纯 CPU 上的"稳跑"闸门 —— 模型越大, 训练切口越窄、单步序列越短、嵌套思考步数越省,
#   确保只靠 CPU 也不溢出内存、不把一轮对话拖到几十秒。默认 2.4B 档时这三项维持原值不变。
if MODEL_PARAMS >= 3_000_000_000:
    TRAIN_LAST_BLOCKS = 1     # 3B/4B: 只训最后 1 层 FFN 的低秩切口, fp32 主权重不膨胀
    TRAIN_MAX_SEQ = 32        # 3B/4B: 单步训练序列截到 32, 反向传播的瞬时内存不翻倍
    TRAIN_STEPS_PER_TURN = 1  # 3B/4B: 每轮 1 步梯度更新, 纯 CPU 也感觉不到卡
    NEST_TOK = 8              # 3B/4B: 嵌套思考词元收窄, 两层思考总耗时仍可控
SCALE_NOTE = ("尺度天花板 = {} · 实选 {} · 参数量 ≈ {:.2f}B".format(
    _SCALE_CAP, MODEL_TIER, MODEL_PARAMS / 1e9))


class AdamW:
    """v1.7 Alpha: 纯原生 AdamW 优化器 (一阶/二阶动量 + 偏差校正 + 解耦权重衰减)。
    参数是设备端 fp32 主权重; 与 int8 量化权重分离, 训练在 fp32 上做, 前向照旧省内存。"""
    def __init__(self, lr=TRAIN_LR, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(wd)
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, param, grad):
        """param: 原地更新的 fp32 数组; grad: 同形状梯度. 返回梯度范数."""
        if grad is None:
            return 0.0
        g = XP.asarray(grad, dtype=XP.float32)
        key = id(param)
        if key not in self.m:
            self.m[key] = XP.zeros_like(param)
            self.v[key] = XP.zeros_like(param)
        self.t += 1
        m, v = self.m[key], self.v[key]
        m *= self.b1
        m += (1.0 - self.b1) * g
        v *= self.b2
        v += (1.0 - self.b2) * (g * g)
        mh = m / (1.0 - self.b1 ** self.t)
        vh = v / (1.0 - self.b2 ** self.t)
        upd = mh / (XP.sqrt(vh) + self.eps)
        if self.wd:
            upd = upd + self.wd * param
        param -= self.lr * upd
        return float(XP.linalg.norm(g))


class TrainBank:
    """v1.7 Alpha: 可训练权重集合。把关键子模块从"固定常量"升级为"fp32 主权重 + 梯度"。
    保存训练步数 / 损失曲线, 供深度思考面板与冒烟测试展示"确实在学"。"""
    def __init__(self, lr=TRAIN_LR):
        self.opt = AdamW(lr)
        self.params = []          # [(name, fp32 数组)]
        self.steps = 0
        self.last_loss = None
        self.loss_hist = []
        self.last_gnorm = 0.0

    def register(self, name, arr):
        self.params.append((name, arr))
        return arr

    def set_lr(self, lr):
        self.opt.lr = float(lr)

    def trainable_params(self):
        return int(sum(int(p.size) for _n, p in self.params))

    def stats(self):
        return {"steps": self.steps, "loss": self.last_loss,
                "params": self.trainable_params(),
                "lr": self.opt.lr, "gnorm": self.last_gnorm}


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
        # ============================================================
        # v1.7 Alpha 核心升级: 权重不再固定 —— 可训练 fp32 主权重 + AdamW + 反向传播
        #   · out_adapter (d×d, 残差式): 直接改写"最后一跳"的词分布
        #   · gate (= _ffn_norm, d 维): 逐维调制深层表征
        #   · 最后 TRAIN_LAST_BLOCKS 层 FFN 的低秩修正 (A: d_ff×r, B: d×r): 真梯度回传到 FFN 通路
        #   只训练这些"小切口" fp32 主权重; int8 常驻大权重仍只做前向 → 显存/内存不涨, 却真的在学。
        # ============================================================
        self.bank = TrainBank(TRAIN_LR)
        self.out_adapter = XP.zeros((d_model, d_model), dtype=XP.float32)
        self.ffn_lr = {}
        self._r = max(2, int(TRAIN_LORA_RANK))
        self.train_layers = list(range(max(0, n_layers - TRAIN_LAST_BLOCKS), n_layers))
        for _li in self.train_layers:
            _dff = d_model * ffn_ratio
            self.ffn_lr[_li] = {"A": rng.normal(0, 0.02, (_dff, self._r)),
                                "B": XP.zeros((d_model, self._r), dtype=XP.float32)}
        if HAS_GPU:
            self.out_adapter = _to_dev(self.out_adapter)
            for _li in self.train_layers:
                self.ffn_lr[_li] = {"A": _to_dev(self.ffn_lr[_li]["A"]),
                                    "B": _to_dev(self.ffn_lr[_li]["B"])}
        self.bank.register("gate", self._ffn_norm)
        self.bank.register("out_adapter", self.out_adapter)
        for _li in self.train_layers:
            self.bank.register("ffnA{}".format(_li), self.ffn_lr[_li]["A"])
            self.bank.register("ffnB{}".format(_li), self.ffn_lr[_li]["B"])
        self.train_steps_total = 0
        self.train_loss_hist = []
        self.train_loaded = 0
        self.load_train_state()

    # ------------------------------------------------------------------
    # v1.7 Alpha: 可训练前向 / 反向传播
    # ------------------------------------------------------------------
    def _block_forward_train(self, blk, i, x):
        """可训练层的前向(与冻结层同构, 额外叠加低秩 FFN 修正)。"""
        a, attn = blk.attn.forward(blk.norm1.forward(x))
        x1 = x + a
        h = blk.norm2.forward(x1)
        W1, W2 = blk.W1.val(), blk.W2.val()
        act = _gelu(h @ W1 + blk.b1)
        f = act @ W2 + blk.b2
        lr = self.ffn_lr.get(i)
        if lr is not None:
            f = f + (act @ lr["A"]) @ lr["B"].T
        return x1 + f, attn

    def _train_forward(self, ids):
        """v1.7 Alpha: 训练专用前向 —— 缓存反向所需中间量(fp32)。
        返回 (x, cache): x 为 (seq, d) 的末层隐状态。"""
        seq = len(ids)
        embed_f = self.embed.val()
        x = embed_f[ids] + _sinusoid(seq, self.d_model)
        cache = {"seq": seq, "layers": {}}
        for i, blk in enumerate(self.blocks):
            if i in self.ffn_lr:
                a, _attn = blk.attn.forward(blk.norm1.forward(x))
                x1 = x + a
                h = blk.norm2.forward(x1)
                W1, W2 = blk.W1.val(), blk.W2.val()
                g = h @ W1 + blk.b1
                act = _gelu(g)
                lr = self.ffn_lr[i]
                f = act @ W2 + blk.b2 + (act @ lr["A"]) @ lr["B"].T
                cache["layers"][i] = {"x_in": x, "x1": x1, "h": h, "g": g, "act": act,
                                      "W1": W1, "W2": W2, "gamma": blk.norm2.gamma}
                x = x1 + f
            else:
                x, _ = blk.forward(x)
        return x, cache

    def train_step(self, ids, targets):
        """v1.7 Alpha: 一次真实的反向传播 + AdamW 更新。
          ids/targets: 等长的 token id 序列(错位一个即 next-token 监督)。
          返回 {"loss":..., "gnorm":...} 或 None(关闭/无数据)。"""
        if not TRAIN_ENABLED or not self.bank or len(ids) < 2:
            return None
        ids = [int(t) for t in ids[-TRAIN_MAX_SEQ:]]
        targets = [int(t) for t in targets[-TRAIN_MAX_SEQ:]]
        x, cache = self._train_forward(ids)
        seq = x.shape[0]
        # —— 输出头(可训练 gate + 残差适配器) ——
        final_all = x * (1.0 + 0.02 * self._ffn_norm)          # (seq, d)
        Ada = self.out_adapter
        z = final_all + final_all @ Ada                        # 残差式适配
        Wout = self.embed.val().T                              # 输出↔输入共享(冻结)
        logits = z @ Wout + self.b_out                         # (seq, V)
        probs = _softmax(logits, -1)
        tgt = XP.asarray(targets, dtype=XP.int64)
        rows = XP.arange(seq)
        eps = 1e-9
        p_t = XP.clip(probs[rows, tgt], eps, 1.0)
        loss = float(-XP.log(p_t).mean())
        # —— 反向: dL/dlogits = probs - onehot ——
        dlogits = probs
        dlogits[rows, tgt] -= 1.0
        dlogits /= float(seq)
        dz = dlogits @ Wout.T                                  # (seq, d)
        dAda = final_all.T @ dz                                # (d, d)
        dfinal = dz + dz @ Ada.T
        dgate = 0.02 * XP.sum(x * dfinal, axis=0)              # (d,)
        dx = dfinal * (1.0 + 0.02 * self._ffn_norm)            # (seq, d)
        grads = {"out_adapter": dAda, "gate": dgate}
        # —— 反向穿过可训练层的 FFN 通路(注意力冻结, 残差直通) ——
        for i in reversed(self.train_layers):
            c = cache["layers"].get(i)
            if c is None:
                continue
            act, x1, x_in = c["act"], c["x1"], c["x_in"]
            lr = self.ffn_lr[i]
            dx_out = dx
            dact = dx_out @ c["W2"].T + (dx_out @ lr["B"]) @ lr["A"].T
            dA = act.T @ (dx_out @ lr["B"])                    # (d, r) 与 A 同形
            dB = dx_out.T @ (act @ lr["A"])                    # (d, r) 与 B 同形(v1.7 修复: 原先转置反了)
            dg = dact * _gelu_grad(c["g"])
            grads["ffnA{}".format(i)] = dA
            grads["ffnB{}".format(i)] = dB
            dh = dg @ c["W1"].T
            dx = dx_out + _ln_backward(dh, x1, c["gamma"])
        # —— AdamW 更新(每个参数一个 step) ——
        gnorm = 0.0
        for name, arr in self.bank.params:
            g = grads.get(name)
            if g is None:
                continue
            gn = float(np.linalg.norm(np.asarray(_to_host(g), dtype=np.float64)))
            gnorm += gn ** 2
            if TRAIN_CLIP > 0 and gn > TRAIN_CLIP:          # v1.7: 梯度裁剪, 训练更稳
                g = g * (TRAIN_CLIP / (gn + 1e-9))
            self.bank.opt.step(arr, g)
        gnorm = float(np.sqrt(gnorm))
        self.bank.steps += 1
        self.bank.last_loss = loss
        self.bank.last_gnorm = gnorm
        self.bank.loss_hist.append(round(loss, 4))
        if len(self.bank.loss_hist) > 200:
            self.bank.loss_hist = self.bank.loss_hist[-200:]
        self.train_steps_total += 1
        return {"loss": loss, "gnorm": gnorm, "steps": self.bank.steps}

    def train_stats(self):
        st = self.bank.stats() if self.bank else {}
        st["total"] = self.train_steps_total
        st["hist"] = self.bank.loss_hist[-8:] if self.bank else []
        return st

    def _train_path(self):
        return "xiaofang_train_{}.npz".format(VERSION.split()[0].replace(".", ""))

    def dump_train_state(self):
        """v1.7 Alpha: 把训练成果(主权重)落盘, 下次启动接着学 —— 越用越准。"""
        if not self.bank:
            return False
        try:
            payload = {"__meta__": np.asarray([self.bank.steps, len(self.bank.loss_hist)],
                                              dtype=np.float64)}
            for name, arr in self.bank.params:
                payload[name] = _to_host(arr)
            payload["__loss__"] = np.asarray(self.bank.loss_hist[-64:] or [0.0], dtype=np.float64)
            np.savez_compressed(self._train_path(), **payload)
            return True
        except Exception:
            return False

    def load_train_state(self):
        """v1.7 Alpha: 启动时续上历史训练权重(有则加载, 无则用随机初始化)。"""
        try:
            import os as _os
            p = self._train_path()
            if not _os.path.exists(p):
                return 0
            z = np.load(p, allow_pickle=False)
            got = 0
            for name, arr in self.bank.params:
                if name in z and tuple(z[name].shape) == tuple(arr.shape):
                    if HAS_GPU:
                        arr[...] = _to_dev(XP.asarray(z[name], dtype=XP.float32))
                    else:
                        arr[...] = XP.asarray(z[name], dtype=XP.float32)
                    got += 1
            if "__meta__" in z:
                self.bank.steps = int(z["__meta__"][0])
                self.train_steps_total = self.bank.steps
            if "__loss__" in z:
                self.bank.loss_hist = [float(v) for v in z["__loss__"]]
                if self.bank.loss_hist:
                    self.bank.last_loss = self.bank.loss_hist[-1]
            self.train_loaded = got
            return got
        except Exception:
            return 0

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
            if i in self.ffn_lr:                       # v1.7: 可训练层走"带低秩修正"的前向
                x, attn = self._block_forward_train(blk, i, x)
            else:
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
        if getattr(self, "out_adapter", None) is not None:
            final = final + final @ self.out_adapter   # v1.7: 可训练残差适配器(训练后就生效)
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
        if core:                                  # ③ 有实体 → 直接给实质答复(不甩"请再问我一遍"的空话)
            seed = self._seed_answer(core)
            if not seed.endswith(("。", "！", "？")):
                seed += "。"
            return "关于「{}」：{}".format(core, seed)
        # ④ 完全虚指 → 温和澄清, 避免答非所问
        return "这句我没抓到具体想问的点，你补一句场景或换个说法，我立刻贴着你说的答。"

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


# ============================================================
# v1.7 Alpha: 符号微积分引擎 (求导 / 积分 / 一阶微分方程)
#   纯自研, 不走数值逼近: 词法 → 递归下降语法 → 符号 AST → 符号微分/积分 → 化简 → 排版
#   覆盖: 多项式与负幂、sin cos tan asin acos atan sinh cosh、exp/ln/log/sqrt/abs、
#         积法则、商法则、链式法则; 微分方程覆盖"直接积分型 / 可分离变量型 / 指数增长型"
# ============================================================
_SYM_FNS = ("sin", "cos", "tan", "asin", "acos", "atan",
            "exp", "ln", "log", "sqrt", "abs", "sinh", "cosh")
_SYM_CONSTS = {"π": math.pi, "e": math.e}


def _sym_numstr(v):
    try:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
    except Exception:
        pass
    return ("%.6f" % v).rstrip("0").rstrip(".") or "0"


def _sym_tokens(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            toks.append(("num", float(s[i:j])))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            w = s[i:j].lower()
            toks.append(("fn", w) if w in _SYM_FNS else ("var", w))
            i = j
            continue
        if c == "π":
            toks.append(("const", "π"))
            i += 1
            continue
        if c in "+-*/^":
            toks.append(("op", c))
            i += 1
            continue
        if c in "()":
            toks.append(("par", c))
            i += 1
            continue
        if c in "×·":
            toks.append(("op", "*"))
            i += 1
            continue
        if c == "÷":
            toks.append(("op", "/"))
            i += 1
            continue
        raise ValueError("非法字符: " + c)
    if not toks:
        raise ValueError("空表达式")
    return toks


class _SymParser:
    """递归下降语法分析: 表达式 → 项 → 一元 → 幂 → 原子, 支持隐式乘法 2x / 3sin(x)"""

    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def _take(self, kind=None, val=None):
        k, v = self._peek()
        if kind is not None and k != kind:
            raise ValueError("语法错误")
        if val is not None and v != val:
            raise ValueError("语法错误")
        self.i += 1
        return v

    def parse(self):
        e = self._expr()
        if self.i != len(self.t):
            raise ValueError("表达式不完整")
        return e

    def _expr(self):
        e = self._term()
        while True:
            k, v = self._peek()
            if k == "op" and v in "+-":
                self._take()
                e = (("add" if v == "+" else "sub"), e, self._term())
            else:
                return e

    def _term(self):
        e = self._unary()
        while True:
            k, v = self._peek()
            if k == "op" and v in "*/":
                self._take()
                e = (("mul" if v == "*" else "div"), e, self._unary())
            elif k in ("num", "var", "const", "fn") or (k == "par" and v == "("):
                e = ("mul", e, self._unary())
            else:
                return e

    def _unary(self):
        k, v = self._peek()
        if k == "op" and v == "-":
            self._take()
            return ("neg", self._unary())
        if k == "op" and v == "+":
            self._take()
            return self._unary()
        return self._power()

    def _power(self):
        b = self._atom()
        k, v = self._peek()
        if k == "op" and v == "^":
            self._take()
            return ("pow", b, self._unary())
        return b

    def _atom(self):
        k, v = self._peek()
        if k == "num":
            self._take()
            return ("num", v)
        if k == "const":
            self._take()
            return ("const", v, _SYM_CONSTS[v])
        if k == "var":
            self._take()
            if v in _SYM_CONSTS:
                return ("const", v, _SYM_CONSTS[v])
            return ("var", v)
        if k == "fn":
            self._take()
            k2, v2 = self._peek()
            if k2 == "par" and v2 == "(":
                self._take("par", "(")
                a = self._expr()
                self._take("par", ")")
            else:
                a = self._atom()
            return ("fn", v, a)
        if k == "par" and v == "(":
            self._take()
            a = self._expr()
            self._take("par", ")")
            return a
        raise ValueError("无法解析的数学式")


def _sym_has(n, name):
    k = n[0]
    if k == "var":
        return n[1] == name
    if k in ("num", "const"):
        return False
    if k == "fn":
        return _sym_has(n[2], name)
    if k == "neg":
        return _sym_has(n[1], name)
    if k in ("add", "sub", "mul", "div", "pow"):
        return _sym_has(n[1], name) or _sym_has(n[2], name)
    return False


def _sym_const_node(n):
    return n[0] in ("num", "const")


def _sym_numval(n):
    if n[0] == "num":
        return n[1]
    if n[0] == "const":
        return n[2]
    return None


def _sym_simplify(n):
    k = n[0]
    if k in ("num", "const", "var"):
        return n
    if k == "neg":
        a = _sym_simplify(n[1])
        if a[0] == "num":
            return ("num", -a[1])
        if a[0] == "neg":
            return a[1]
        return ("neg", a)
    if k in ("add", "sub"):
        a, b = _sym_simplify(n[1]), _sym_simplify(n[2])
        if a[0] == "num" and b[0] == "num":
            return ("num", a[1] + b[1] if k == "add" else a[1] - b[1])
        if k == "add":
            if a[0] == "num" and abs(a[1]) < 1e-15:
                return b
            if b[0] == "num" and abs(b[1]) < 1e-15:
                return a
        else:
            if b[0] == "num" and abs(b[1]) < 1e-15:
                return a
            if a[0] == "num" and abs(a[1]) < 1e-15:
                return ("neg", b)
        # (a-b) 且两者完全相同 → 0
        if k == "sub" and a == b:
            return ("num", 0.0)
        return (k, a, b)
    if k == "mul":
        a, b = _sym_simplify(n[1]), _sym_simplify(n[2])
        if a[0] == "num" and b[0] == "num":
            return ("num", a[1] * b[1])
        if (a[0] == "num" and abs(a[1]) < 1e-15) or (b[0] == "num" and abs(b[1]) < 1e-15):
            return ("num", 0.0)
        if b[0] == "num" and abs(b[1] - 1.0) < 1e-15:
            return a
        if a[0] == "num" and abs(a[1] - 1.0) < 1e-15:
            return b
        if a[0] == "neg":
            return _sym_simplify(("neg", ("mul", a[1], b)))
        if b[0] == "neg":
            return _sym_simplify(("neg", ("mul", a, b[1])))
        if a[0] == "num" and abs(a[1] + 1.0) < 1e-15:
            return ("neg", b)
        if b[0] == "num" and abs(b[1] + 1.0) < 1e-15:
            return ("neg", a)
        if a[0] != "num" and b[0] == "num":
            a, b = b, a
        # v1.7 Alpha: c·(u/d) → (c·u)/d, 便于常数因子约分 (如 2·(x^2/2) → x^2)
        if a[0] == "num" and b[0] == "div":
            return _sym_simplify(("div", ("mul", a, b[1]), b[2]))
        return ("mul", a, b)
    if k == "div":
        a, b = _sym_simplify(n[1]), _sym_simplify(n[2])
        if b[0] == "num" and abs(b[1]) < 1e-15:
            return ("div", a, b)
        if a[0] == "num" and b[0] == "num":
            return ("num", a[1] / b[1])
        if a[0] == "num" and abs(a[1]) < 1e-15:
            return ("num", 0.0)
        if b[0] == "num" and abs(b[1] - 1.0) < 1e-15:
            return a
        if a == b:
            return ("num", 1.0)
        # v1.7 Alpha: 常数因子约分 (c·u)/d → (c/d)·u, 如 2*x^2/2 → x^2
        if a[0] == "mul" and b[0] == "num" and abs(b[1]) > 1e-15:
            if a[1][0] == "num":
                q, u = a[1][1], a[2]
            elif a[2][0] == "num":
                q, u = a[2][1], a[1]
            else:
                q = u = None
            if u is not None:
                return _sym_simplify(("mul", ("num", q / b[1]), u))
        # v1.7 Alpha: 叠分数 a / (b/c) → (a·c)/b
        if b[0] == "div":
            return _sym_simplify(("div", ("mul", a, b[2]), b[1]))
        return ("div", a, b)
    if k == "pow":
        a, b = _sym_simplify(n[1]), _sym_simplify(n[2])
        if b[0] == "num":
            if abs(b[1]) < 1e-15:
                return ("num", 1.0)
            if abs(b[1] - 1.0) < 1e-15:
                return a
        if a[0] == "num" and b[0] == "num":
            try:
                return ("num", float(a[1] ** b[1]))
            except Exception:
                return ("pow", a, b)
        return ("pow", a, b)
    if k == "fn":
        a = _sym_simplify(n[2])
        if n[1] == "ln" and a[0] == "const" and a[1] == "e":
            return ("num", 1.0)
        if a[0] == "neg":
            if n[1] == "sin":
                return ("neg", ("fn", "sin", a[1]))
            if n[1] == "cos":
                return ("fn", "cos", a[1])
        return ("fn", n[1], a)
    return n


def _sym_str(n, prec=0):
    k = n[0]
    if k == "num":
        return _sym_numstr(n[1])
    if k == "const":
        return n[1]
    if k == "var":
        return n[1]
    if k == "neg":
        s = "-" + _sym_str(n[1], 2)
        return "(" + s + ")" if prec > 2 else s
    if k in ("add", "sub"):
        s = _sym_str(n[1], 1) + (" + " if k == "add" else " - ") + _sym_str(n[2], 2)
        return "(" + s + ")" if prec > 1 else s
    if k == "mul":
        s = _sym_str(n[1], 2) + "*" + _sym_str(n[2], 2)
        return "(" + s + ")" if prec > 2 else s
    if k == "div":
        s = _sym_str(n[1], 2) + "/" + _sym_str(n[2], 3)
        return "(" + s + ")" if prec > 2 else s
    if k == "pow":
        s = _sym_str(n[1], 4) + "^" + _sym_str(n[2], 4)
        return "(" + s + ")" if prec > 3 else s
    if k == "fn":
        return n[1] + "(" + _sym_str(n[2], 0) + ")"
    return "?"


def _sym_fn_deriv(f, u):
    if f == "sin":
        return ("fn", "cos", u)
    if f == "cos":
        return ("neg", ("fn", "sin", u))
    if f == "tan":
        return ("div", ("num", 1.0), ("pow", ("fn", "cos", u), ("num", 2.0)))
    if f == "exp":
        return ("fn", "exp", u)
    if f in ("ln", "log"):
        return ("div", ("num", 1.0), u)
    if f == "sqrt":
        return ("div", ("num", 1.0), ("mul", ("num", 2.0), ("fn", "sqrt", u)))
    if f == "asin":
        return ("div", ("num", 1.0),
                ("fn", "sqrt", ("sub", ("num", 1.0), ("pow", u, ("num", 2.0)))))
    if f == "acos":
        return ("neg", ("div", ("num", 1.0),
                        ("fn", "sqrt", ("sub", ("num", 1.0), ("pow", u, ("num", 2.0))))))
    if f == "atan":
        return ("div", ("num", 1.0), ("add", ("num", 1.0), ("pow", u, ("num", 2.0))))
    if f == "sinh":
        return ("fn", "cosh", u)
    if f == "cosh":
        return ("fn", "sinh", u)
    if f == "abs":
        return ("div", u, ("fn", "abs", u))
    return None


def _sym_diff(n, var="x"):
    k = n[0]
    if k in ("num", "const"):
        return ("num", 0.0)
    if k == "var":
        return ("num", 1.0 if n[1] == var else 0.0)
    if k == "neg":
        return ("neg", _sym_diff(n[1], var))
    if k in ("add", "sub"):
        return (k, _sym_diff(n[1], var), _sym_diff(n[2], var))
    if k == "mul":
        a, b = n[1], n[2]
        return ("add", ("mul", _sym_diff(a, var), b), ("mul", a, _sym_diff(b, var)))
    if k == "div":
        a, b = n[1], n[2]
        return ("div",
                ("sub", ("mul", _sym_diff(a, var), b), ("mul", a, _sym_diff(b, var))),
                ("pow", b, ("num", 2.0)))
    if k == "pow":
        a, b = n[1], n[2]
        if b[0] == "num":
            return ("mul", ("mul", b, ("pow", a, ("num", b[1] - 1.0))), _sym_diff(a, var))
        if _sym_const_node(a):
            return ("mul", ("mul", n, ("fn", "ln", a)), _sym_diff(b, var))
        return ("mul", n, ("add",
                           ("mul", _sym_diff(b, var), ("fn", "ln", a)),
                           ("div", ("mul", b, _sym_diff(a, var)), a)))
    if k == "fn":
        d = _sym_fn_deriv(n[1], n[2])
        if d is None:
            return ("num", 0.0)
        return ("mul", d, _sym_diff(n[2], var))
    return ("num", 0.0)


def _sym_linear(u, var):
    """u 是 var 的一次式时返回 (a, b) 满足 u = a*var + b; 否则 None"""
    if u[0] == "var":
        return (1.0, 0.0) if u[1] == var else None
    if u[0] == "num":
        return (0.0, u[1])
    if u[0] == "const":
        return (0.0, u[2])
    if u[0] == "neg":
        r = _sym_linear(u[1], var)
        return (-r[0], -r[1]) if r else None
    if u[0] in ("add", "sub"):
        la, lb = _sym_linear(u[1], var), _sym_linear(u[2], var)
        if not la or not lb:
            return None
        s = 1.0 if u[0] == "add" else -1.0
        return (la[0] + s * lb[0], la[1] + s * lb[1])
    if u[0] == "mul":
        la, lb = _sym_linear(u[1], var), _sym_linear(u[2], var)
        if not la or not lb:
            return None
        if abs(la[0]) < 1e-12 and abs(lb[0]) > 1e-12:
            return (la[1] * lb[0], la[1] * lb[1])
        if abs(lb[0]) < 1e-12 and abs(la[0]) > 1e-12:
            return (lb[1] * la[0], lb[1] * la[1])
        if abs(la[0]) < 1e-12 and abs(lb[0]) < 1e-12:
            return (0.0, la[1] * lb[1])
        return None
    if u[0] == "div":
        la, lb = _sym_linear(u[1], var), _sym_linear(u[2], var)
        if la and lb and abs(lb[0]) < 1e-12 and abs(lb[1]) > 1e-12:
            return (la[0] / lb[1], la[1] / lb[1])
        return None
    return None


def _sym_integrate(n, var="x"):
    """不定积分(不含 +C); 无法积出返回 None"""
    k = n[0]
    if k in ("num", "const"):
        return ("mul", n, ("var", var))
    if k == "var":
        if n[1] != var:
            return ("mul", n, ("var", var))
        return ("div", ("pow", ("var", var), ("num", 2.0)), ("num", 2.0))
    if k == "neg":
        r = _sym_integrate(n[1], var)
        return ("neg", r) if r else None
    if k in ("add", "sub"):
        a = _sym_integrate(n[1], var)
        b = _sym_integrate(n[2], var)
        return (k, a, b) if (a is not None and b is not None) else None
    if k == "mul":
        a, b = n[1], n[2]
        if _sym_const_node(a):
            r = _sym_integrate(b, var)
            return ("mul", a, r) if r else None
        if _sym_const_node(b):
            r = _sym_integrate(a, var)
            return ("mul", b, r) if r else None
        return None
    if k == "div":
        a, b = n[1], n[2]
        if _sym_const_node(a):
            if b[0] == "var" and b[1] == var:
                return ("mul", a, ("fn", "ln", ("fn", "abs", b)))
            if b[0] == "pow" and b[1] == ("var", var) and b[2][0] == "num" and abs(b[2][1] + 1.0) > 1e-12:
                return ("mul", a, _sym_integrate(b, var))
            return None
        return None
    if k == "pow":
        base, ex = n[1], n[2]
        if base[0] == "var" and base[1] == var and ex[0] == "num":
            p = ex[1]
            if abs(p + 1.0) < 1e-12:
                return ("fn", "ln", ("fn", "abs", base))
            return ("div", ("pow", base, ("num", p + 1.0)), ("num", p + 1.0))
        if _sym_const_node(base):
            la = _sym_linear(ex, var)
            if la and abs(la[0]) > 1e-12:
                return ("div", n, ("mul", ("fn", "ln", base), ("num", la[0])))
        return None
    if k == "fn":
        f, u = n[1], n[2]
        la = _sym_linear(u, var)
        if not la or abs(la[0]) < 1e-12:
            return None
        a = la[0]
        if f == "sin":
            return ("mul", ("num", -1.0 / a), ("fn", "cos", u))
        if f == "cos":
            return ("mul", ("num", 1.0 / a), ("fn", "sin", u))
        if f == "exp":
            return ("mul", ("num", 1.0 / a), ("fn", "exp", u))
        if f == "sinh":
            return ("mul", ("num", 1.0 / a), ("fn", "cosh", u))
        if f == "cosh":
            return ("mul", ("num", 1.0 / a), ("fn", "sinh", u))
        if f in ("ln", "log"):
            # ∫ln(u): 分部积分 → u*ln(u) - u
            return ("sub", ("mul", u, ("fn", "ln", u)), u)
        return None
    return None


def _sym_eval(n, var, val):
    k = n[0]
    if k == "num":
        return float(n[1])
    if k == "const":
        return float(n[2])
    if k == "var":
        if n[1] == var:
            return float(val)
        raise ValueError("未知变量 " + n[1])
    if k == "neg":
        return -_sym_eval(n[1], var, val)
    if k == "add":
        return _sym_eval(n[1], var, val) + _sym_eval(n[2], var, val)
    if k == "sub":
        return _sym_eval(n[1], var, val) - _sym_eval(n[2], var, val)
    if k == "mul":
        return _sym_eval(n[1], var, val) * _sym_eval(n[2], var, val)
    if k == "div":
        return _sym_eval(n[1], var, val) / _sym_eval(n[2], var, val)
    if k == "pow":
        return _sym_eval(n[1], var, val) ** _sym_eval(n[2], var, val)
    if k == "fn":
        u = _sym_eval(n[2], var, val)
        f = n[1]
        m = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
             "ln": math.log, "log": math.log, "sqrt": math.sqrt, "abs": abs,
             "asin": math.asin, "acos": math.acos, "atan": math.atan,
             "sinh": math.sinh, "cosh": math.cosh}
        if f in m:
            return float(m[f](u))
    raise ValueError("无法求值")


def _sym_separate(n, yname):
    """把 dy/dx = f(x)*g(y) 拆成 (f_x, g_y); 拆不开返回 None"""
    # 右端只含 y(不含 x) → 也可分离: g(y) = 整体, f(x) = 1
    if _sym_has(n, yname) and not _sym_has(n, "x"):
        return (("num", 1.0), n)
    if n[0] == "mul":
        a, b = n[1], n[2]
        ha, hb = _sym_has(a, yname), _sym_has(b, yname)
        if ha and not hb:
            return (b, a)
        if hb and not ha:
            return (a, b)
        return None
    if n[0] == "div":
        a, b = n[1], n[2]
        if _sym_has(b, yname) and not _sym_has(a, yname):
            return (a, _sym_simplify(("div", ("num", 1.0), b)))
        return None
    return None


def _sym_ln_y_coef(g):
    """若 ∫dy/g 为 c*ln|y| 形式, 返回 c; 否则 None"""
    if g[0] == "var" and g[1] == "y":
        return 1.0
    if g[0] == "mul":
        a, b = g[1], g[2]
        if _sym_const_node(a) and b[0] == "var" and b[1] == "y":
            c = _sym_numval(a)
            return (1.0 / c) if c else None
        if _sym_const_node(b) and a[0] == "var" and a[1] == "y":
            c = _sym_numval(b)
            return (1.0 / c) if c else None
    return None


class MathResponder:
    # 支持 + - * / % ( ) 与中文"加减乘除"、平方/立方; 安全解析(非 eval)
    REP = {"×": "*", "÷": "/", "＋": "+", "－": "-", "＝": "=", "（": "(", "）": ")", "％": "%"}

    def detect(self, raw):
        if not isinstance(raw, str):
            return False
        # v1.5 Alpha: 数学 AST 触发 —— 含函数/根号/常量且带数字
        if re.search(r"(?:sin|cos|tan|sqrt|log|ln|abs|floor|ceil|π|pi|根号|开方|平方根|取整)", raw, re.I) \
                and re.search(r"\d", raw):
            return True
        if re.search(r"\d\s*[\+\-*/%×÷]\s*\d", raw):
            return True
        if re.search(r"[\d一二两三四五六七八九十百千万]+(?:加|减|乘|除|乘以|除以|的平方|的立方|等于多少|等于|求值)", raw):
            return True
        if any(w in raw for w in ["算一下", "计算", "数学题", "求和", "求值"]):
            return True
        # v1.5 正式版: 多元一次方程组 / 矩阵运算 也判为数学
        if any(w in raw for w in ["方程组", "三元", "二元", "元一次"]):
            return True
        if re.search(r"\[\[", raw) or any(w in raw for w in ["矩阵", "行列式", "行列"]):
            return True
        # v1.7 Alpha: 微积分(求导/积分/微分方程)判为数学 —— 交给符号引擎(_calc_route)。
        if any(w in raw for w in ["求导", "导数", "微分方程", "微分", "积分", "∫", "原函数", "不定积分",
                                  "定积分", "二阶导", "偏导", "d/dx", "dy/dx"]):
            return True
        if re.search(r"d\s*[y2-9]?\s*/\s*d\s*[x2-9]", raw):
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
        s = re.sub(r"(?<![a-zA-Z])[a-df-zA-DF-Z](?=\d)", lambda mo: mo.group(0) + "*", s)
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

    # ---------- v1.5 正式版: 多元一次方程组 + 矩阵运算 (高斯消元 + 安全行列式) ----------
    def _parse_linear_eq(self, s):
        """把一条 'lhs=rhs' 归一成 {变量:系数} 与常数项 b, 满足 Σ系数·x = b。"""
        s = (s or "").strip().replace("·", "").replace("×", "").replace("*", "").replace(" ", "")
        if "=" in s:
            lh, rh = s.split("=", 1)
        else:
            lh, rh = s, "0"
        def terms(expr):
            cd, con = {}, 0.0
            for p in re.findall(r"[+-]?[^+-]+", expr):
                p = p.strip()
                if not p:
                    continue
                neg = p[0] == "-"
                core = p.lstrip("+-")
                m = re.fullmatch(r"(\d+(?:\.\d+)?)?([a-zA-Z])", core)
                if m:
                    num, var = m.group(1), m.group(2)
                    v = float(num) if num else 1.0
                    cd[var] = cd.get(var, 0.0) + (-v if neg else v)
                elif re.fullmatch(r"\d+(?:\.\d+)?", core):
                    con += -float(core) if neg else float(core)
            return cd, con
        cl, kl = terms(lh)
        cr, kr = terms(rh)
        allv = set(cl) | set(cr)
        return {v: cl.get(v, 0.0) - cr.get(v, 0.0) for v in allv}, -(kl - kr)

    @staticmethod
    def _gauss(A, b):
        n = len(A)
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-12:
                return None
            M[col], M[piv] = M[piv], M[col]
            v = M[col][col]
            for r in range(n):
                if r == col or abs(M[r][col]) < 1e-12:
                    continue
                f = M[r][col] / v
                for c in range(n + 1):
                    M[r][c] -= f * M[col][c]
        return [M[i][n] / M[i][i] for i in range(n)]

    def _solve_linear(self, raw):
        eqs = re.findall(r"([0-9a-zA-Z+\-*/·×\s.]+)=([0-9a-zA-Z+\-*/·×\s.]+)", raw or "")
        parsed = []
        for lh, rh in eqs:
            # 只要求含变量字母；系数可能全是隐式1(如 x+y=5)，不能硬性要求左侧带数字
            if not re.search(r"[a-df-zA-DF-Z]", lh + rh):
                continue
            cd, b = self._parse_linear_eq(lh + "=" + rh)
            if cd:
                parsed.append((cd, b))
        if not parsed:
            return None
        allv = []
        for cd, _ in parsed:
            for v in cd:
                if v not in allv:
                    allv.append(v)
        allv.sort(key=lambda v: "xyzabcd".index(v) if v in "xyzabcd" else 9)
        n = len(allv)
        if len(parsed) < n:
            return "🧮 方程组有 {} 个未知数（{}）却只给了 {} 个方程，唯一解不足，请补充条件。".format(
                n, "、".join(allv), len(parsed))
        idx = {v: i for i, v in enumerate(allv)}
        A, b = [], []
        for cd, bb in parsed:
            row = [0.0] * n
            for v, c in cd.items():
                if v in idx:
                    row[idx[v]] = c
            A.append(row)
            b.append(bb)
        x = self._gauss(A, b)
        if x is None:
            return "🧮 方程组系数矩阵退化（无解或有无穷多解），请核对条件。"
        disp = "，".join("**{} = {}**".format(allv[i], self._fmt(x[i])) for i in range(n))
        return "🧮 解 {} 元一次方程组：{}".format(n, disp)

    @staticmethod
    def _parse_matrix(s):
        i = s.find("[")
        if i < 0:
            return None, s
        depth = 0
        j = i
        while j < len(s):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return None, s
        body, rest = s[i + 1:j], s[j + 1:]
        rows = []
        cur = ""
        for ch in body:
            cur += ch
            if ch == "]":
                rows.append(cur)
                cur = ""
        M = []
        for r in rows:
            nums = re.findall(r"-?\d+(?:\.\d+)?", r)
            if nums:
                M.append([float(x) for x in nums])
        return (M, rest) if M else (None, rest)

    @staticmethod
    def _mat_mul(A, B):
        if not A or not B or len(A[0]) != len(B):
            return None
        ra, cb, ca = len(A), len(B[0]), len(A[0])
        C = [[0.0] * cb for _ in range(ra)]
        for i in range(ra):
            for k in range(ca):
                aik = A[i][k]
                for j in range(cb):
                    C[i][j] += aik * B[k][j]
        return C

    def _mat_det(self, A):
        n = len(A)
        if any(len(r) != n for r in A):
            return None
        M = [r[:] for r in A]
        det = 1.0
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-12:
                return 0.0
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
                det = -det
            v = M[col][col]
            det *= v
            for r in range(col + 1, n):
                f = M[r][col] / v
                for c in range(col, n):
                    M[r][c] -= f * M[col][c]
        return det

    def _mat_str(self, M):
        return "\n".join("[ " + ", ".join(self._fmt(x) for x in r) + " ]" for r in M)

    def _mat_compute(self, raw):
        s = re.sub(r"\s", "", raw or "")
        if any(k in raw for k in ["行列式", "行列", "determinant", "det("]) or raw.lower().startswith("det"):
            A, _ = self._parse_matrix(s)
            if A and len(A) == len(A[0]):
                d = self._mat_det(A)
                if d is not None:
                    return "🧮 行列式 det(A) = **{}**".format(self._fmt(d))
        mats = []
        rest = s
        while "[" in rest:
            m, rest = self._parse_matrix(rest)
            if not m:
                break
            mats.append(m)
        if len(mats) < 2:
            return None
        A, B = mats[0], mats[1]
        if any(k in raw for k in ["相加", "加法", "求和"]):
            if len(A) != len(B) or len(A[0]) != len(B[0]):
                return "🧮 矩阵加法要求同形状，两者不同无法相加。"
            C = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
            op = "+"
        else:                      # 无显式词 → 默认矩阵乘法(最常见)
            if any(k in raw for k in ["相乘", "乘积", "乘法", "积", "@"]):
                pass
            C = self._mat_mul(A, B)
            if C is None:
                return "🧮 矩阵乘法形状不匹配：({}×{}) 与 ({}×{}) 无法相乘。".format(
                    len(A), len(A[0]), len(B), len(B[0]))
            op = "×"
        return "🧮 矩阵结果（A{0}B）:\n```\n{1}\n```".format(op, self._mat_str(C))

    # ---------- v1.5 Alpha: 数学 AST —— 安全求值, 支持函数/常量/幂/根号(非 eval 字符串执行) ----------
    _FUNC_CN = {"根号": "sqrt(", "平方根": "sqrt(", "正弦": "sin(", "余弦": "cos(",
                "正切": "tan(", "自然对数": "ln(", "对数": "log(", "绝对值": "abs(",
                "向下取整": "floor(", "向上取整": "ceil("}

    def _ast_value(self, norm):
        """从规范文本抠出可安全计算的表达式并求值。返回 (表达式, 数值) 或 None。"""
        s = re.split(r"[=＝]", norm)[0]
        # 中文一元前缀: 根号/平方根 → sqrt(  (针对 "根号16" 等)
        s = re.sub(r"平方根|根号", "sqrt(", s)
        # 中文二元运算符 → 符号 (兼容 "9加4"/"2乘根号2"/"10除以2" 等函数或常量参与的情形)
        s = re.sub(r"乘以|乘上", "*", s)
        s = re.sub(r"除以|除去", "/", s)
        s = re.sub(r"加|加上", "+", s)
        s = re.sub(r"减|减去", "-", s)
        s = re.sub(r"乘", "*", s)
        s = re.sub(r"除", "/", s)
        for cn, en in self._FUNC_CN.items():   # 中文函数名→英文 (正弦/余弦/绝对值/取整/对数…)
            s = s.replace(cn, en)
        # 把"sin30"这类函数后无括号的写法补成"sin(30": 只当下一字符是数字/π 且没紧跟 '(' 时
        s = re.sub(r"(?<![0-9a-z])(sin|cos|tan|sqrt|log|ln|abs|floor|ceil)(?!\s*\()(?=[0-9π])",
                   r"\1(", s)
        # π 与上标/次方须先转成 ASCII(pi / **n), 再清中文残留, 否则会被丢
        s = s.replace("π", "pi").replace("^", "**").replace("²", "**2").replace("³", "**3")
        s = re.sub(r"[^\d.()+\-*/%^ a-zA-Z_]+", " ", s)   # 去掉中文残留
        # 扫描连续数学表达式(数字/运算/括号/常量/函数名)
        out, i, started = [], 0, False
        fun = re.compile(r"(?<![0-9a-z])(?:sin|cos|tan|sqrt|log|ln|abs|floor|ceil|pi|e)(?![0-9a-z])")
        while i < len(s):
            ch = s[i]
            if ch.isdigit() or ch in ".()+-*/% ":
                if ch.strip() or started:
                    started = True
                    out.append(ch)
                i += 1
                continue
            m = fun.match(s, i)
            if m:
                started = True
                out.append(m.group(0))
                i = m.end()
                continue
            if started:
                break       # 遇到非数学 token 结束
            i += 1
        expr = "".join(out).strip()
        if not expr or not re.search(r"\d|pi|e|\)", expr):
            return None
        if expr.count("(") > expr.count(")"):        # "根号16"→sqrt(16 补全右括号
            expr += ")" * (expr.count("(") - expr.count(")"))
        try:
            if re.search(r"(?<![0-9a-z])(sin|cos|tan|sqrt|log|ln|abs|floor|ceil|pi|e)(?![0-9a-z])", expr):
                val = self._ast_eval(expr)           # 含函数/常量 → 必须走安全 AST
            else:
                val = _MathParser(expr).parse()      # 纯算术快速路径
        except Exception:
            try:
                val = self._ast_eval(expr)
            except Exception:
                return None
        return (expr, val)

    def _ast_eval(self, expr):
        import ast as _A, math as _m, builtins as _b
        FUNCS = {"sin": _m.sin, "cos": _m.cos, "tan": _m.tan, "sqrt": _m.sqrt,
                 "log": _m.log, "ln": _m.log, "abs": _b.abs,          # math 无 abs → 内建
                 "floor": _m.floor, "ceil": _m.ceil}
        CONSTS = {"pi": _m.pi, "e": _m.e}

        def ev(node):
            if isinstance(node, _A.Constant):
                v = node.value
                if isinstance(v, (int, float)):
                    return float(v)
                raise ValueError("非数字")
            if isinstance(node, _A.Name):
                if node.id in CONSTS:
                    return CONSTS[node.id]
                raise ValueError("未知名")
            if isinstance(node, _A.BinOp):
                a, b = ev(node.left), ev(node.right)
                if isinstance(node.op, _A.Add): return a + b
                if isinstance(node.op, _A.Sub): return a - b
                if isinstance(node.op, _A.Mult): return a * b
                if isinstance(node.op, _A.Div):
                    if abs(b) < 1e-12: raise ZeroDivisionError
                    return a / b
                if isinstance(node.op, _A.FloorDiv):
                    if abs(b) < 1e-12: raise ZeroDivisionError
                    return a // b
                if isinstance(node.op, _A.Mod): return a % b
                if isinstance(node.op, _A.Pow):
                    return a ** b if b == int(b) and b >= 0 else a ** b
                raise ValueError("非法运算")
            if isinstance(node, _A.UnaryOp):
                if isinstance(node.op, _A.USub): return -ev(node.operand)
                if isinstance(node.op, _A.UAdd): return ev(node.operand)
                raise ValueError("非法单目")
            if isinstance(node, _A.Call):
                if isinstance(node.func, _A.Name) and node.func.id in FUNCS:
                    args = [ev(a) for a in node.args]
                    if len(args) != 1:
                        raise ValueError("参数个数")
                    v = args[0]
                    if node.func.id in ("sin", "cos", "tan"):
                        v = v * _m.pi / 180.0      # 日常用法: 三角按度数理解
                    return FUNCS[node.func.id](v)
                raise ValueError("未知函数")
            raise ValueError("非法AST")

        return ev(_A.parse(expr, mode="eval").body)

    def _ast_compute(self, norm):
        got = self._ast_value(norm)
        if got is None:
            return None
        expr, val = got
        if abs(val - round(val)) < 1e-9:
            disp = str(int(round(val)))
        else:
            disp = ("%.6f" % val).rstrip("0").rstrip(".")
        return "🧮 计算 {} 得 **{}**（按数学规则求解）。".format(expr, disp)

    # ============================================================
    # v1.7 Alpha: 微积分路由 (求导 / 积分 / 一阶微分方程)
    # ============================================================
    CALC_KW = ("求导", "导数", "微分", "积分", "∫", "原函数", "dy/dx", "d/dx")

    def _calc_clean(self, s):
        s = s.strip()
        for k, v in self.REP.items():
            s = s.replace(k, v)
        s = s.replace("²", "^2").replace("³", "^3")
        s = s.replace("乘以", "*").replace("除以", "/")
        return s

    def _calc_num(self, t):
        t = (t or "").strip()
        if t in ("π", "pi"):
            return math.pi
        try:
            return float(t)
        except Exception:
            return None

    def _calc_node(self, s):
        """在杂字符串里挑出能解析的最长数学表达式"""
        best_s, best_n = "", None
        seen = set()
        for c in re.findall(r"[0-9a-zA-Z_π+\-*/^(). ]+", s):
            # 逐个剥离两端残留的运算符/括号/空白, 收集所有可解析候选, 取最长可解析者
            #   (直接 strip 会切掉右括号, 把 x^2*sin(x) 弄成 x^2*sin(x → 解析失败)
            cands = [c.strip()]
            t = c.strip()
            while t:
                cands.append(t)
                t = t[1:].strip()
            t = c.strip()
            while t:
                cands.append(t)
                t = t[:-1].strip()
            for t in cands:
                if not t or t in seen:
                    continue
                seen.add(t)
                try:
                    n = _SymParser(_sym_tokens(self._calc_clean(t))).parse()
                except Exception:
                    continue
                if len(t) > len(best_s):
                    best_s, best_n = t, n
        return best_n, best_s

    def _calc_parse(self, raw):
        """从自然语言里提取 (语法树, 自变量, 附加信息)"""
        s, info = raw, {}
        # 微分方程左端符号先摘掉
        s = re.sub(r"(?:dy\s*/\s*dx|d\s*/\s*dx|y\s*['′]|y\s*''\s*)", " ", s, flags=re.I)
        # 积分上下限: "从 a 到 b" / "下限 a 上限 b"
        m = re.search(r"从\s*(-?[\d.]+|π|pi)\s*到\s*(-?[\d.]+|π|pi)", s) or \
            re.search(r"下限\s*(-?[\d.]+|π|pi)\s*上限\s*(-?[\d.]+|π|pi)", s)
        if m:
            info["lo"] = self._calc_num(m.group(1))
            info["hi"] = self._calc_num(m.group(2))
            s = s[:m.start()] + " " + s[m.end():]
        # 自变量: "对t求导"
        var = "x"
        mv = re.search(r"对\s*([a-zA-Z])\s*(?:求导|求微分|积分|求偏导)", s)
        if mv:
            var = mv.group(1).lower()
            s = s[:mv.start()] + " " + s[mv.end():]
        # 去掉中文说明词与残留的微分符号
        s = re.sub(r"[\u4e00-\u9fff]+", " ", s)
        s = re.sub(r"(?i)\b(dx|dy|dt|d)\b", " ", s)
        node, expr = self._calc_node(s)
        if node is None:
            return None, var, info
        info["expr"] = expr
        return node, var, info

    def _calc_derivative(self, raw):
        if not re.search(r"(?:求导|导数|求微分|微分|d\s*/\s*dx)", raw, re.I):
            return None
        if re.search(r"(?:积分|∫|原函数)", raw):
            return None
        if re.search(r"(?:微分方程|dy\s*/\s*dx|y\s*['′])", raw, re.I):
            return None
        node, var, _info = self._calc_parse(raw)
        if node is None:
            return None
        d = _sym_simplify(_sym_diff(node, var))
        f, ds = _sym_str(node), _sym_str(d)
        return ("🧮 **求导**：d/d{0} [ {1} ] = **{2}**\n\n"
                "| 步骤 | 内容 |\n| --- | --- |\n"
                "| 1 | 识别结构：和差 / 积商 / 复合，逐层拆解 |\n"
                "| 2 | 套法则：幂法则、积法则、商法则、链式法则 |\n"
                "| 3 | 合并同类项、约去常数因子，化简到最简 |\n\n"
                "> 结论：f({0}) = {1} ⟹ f'({0}) = {2}").format(var, f, ds)

    def _calc_integral(self, raw):
        if not re.search(r"(?:积分|∫|原函数)", raw):
            return None
        node, var, info = self._calc_parse(raw)
        if node is None:
            return None
        F = _sym_integrate(node, var)
        if F is None:
            return None
        Fs = _sym_simplify(F)
        f, Fstr = _sym_str(node), _sym_str(Fs)
        lo, hi = info.get("lo"), info.get("hi")
        if lo is not None and hi is not None:
            try:
                A, B = _sym_eval(Fs, var, lo), _sym_eval(Fs, var, hi)
            except Exception:
                return None
            v = B - A
            return ("🧮 **定积分**：∫({0}) d{1} ，区间 [{2}, {3}]\n\n"
                    "| 步骤 | 内容 |\n| --- | --- |\n"
                    "| 1 | 求原函数 F({1}) = {4} |\n"
                    "| 2 | 代入上限：F({3}) = {5} |\n"
                    "| 3 | 代入下限：F({2}) = {6} |\n"
                    "| 4 | 上限值 − 下限值 = {7} |\n\n"
                    "> 结论：积分值 = **{7}**").format(
                        f, var, _sym_numstr(lo), _sym_numstr(hi), Fstr,
                        _sym_numstr(B), _sym_numstr(A), _sym_numstr(v))
        return ("🧮 **不定积分**：∫ {0} d{1} = **{2}** + C\n\n"
                "| 步骤 | 内容 |\n| --- | --- |\n"
                "| 1 | 拆成基本积分之和（逐项处理） |\n"
                "| 2 | 幂法则 / 线性换元 / 分部积分 |\n"
                "| 3 | 补上任意常数 C |\n\n"
                "> 结论：∫ {0} d{1} = {2} + C").format(f, var, Fstr)

    def _calc_ode(self, raw):
        if not re.search(r"(?:微分方程|dy\s*/\s*dx|d\s*/\s*dx|y\s*['′])", raw, re.I):
            return None
        m = re.search(r"(?:dy\s*/\s*dx|y\s*['′]|d\s*/\s*dx)\s*[=＝]\s*([^,，。;；\n]+)", raw, re.I)
        if not m:
            return None
        rhs = re.sub(r"[\u4e00-\u9fff]+", " ", m.group(1))
        node, _e = self._calc_node(rhs)
        if node is None:
            return None
        rhs_str = _sym_str(node)
        if not _sym_has(node, "y"):
            F = _sym_integrate(node, "x")
            if F is None:
                return None
            Fs = _sym_simplify(F)
            return ("🧮 **一阶微分方程**：dy/dx = {0}\n\n"
                    "| 步骤 | 内容 |\n| --- | --- |\n"
                    "| 1 | 判定：右端不含 y → 直接积分型 |\n"
                    "| 2 | 两边对 x 积分：y = ∫ {0} dx |\n"
                    "| 3 | 积出原函数，补任意常数 C |\n\n"
                    "> 通解：**y = {1} + C**").format(rhs_str, _sym_str(Fs))
        sep = _sym_separate(node, "y")
        if sep is None:
            return None
        fx, gy = sep
        F = _sym_integrate(fx, "x")
        if F is None:
            return None
        Fs = _sym_simplify(F)
        Fstr = _sym_str(Fs)
        c = _sym_ln_y_coef(gy)
        head = ("🧮 **一阶微分方程**：dy/dx = {0}\n\n"
                "| 步骤 | 内容 |\n| --- | --- |\n"
                "| 1 | 判定：可写成 f(x)·g(y) → 可分离变量型 |\n"
                "| 2 | 分离变量：dy / g(y) = f(x) dx，即 dy / {1} = {2} dx |\n"
                "| 3 | 两边分别积分 |\n").format(rhs_str, _sym_str(gy), _sym_str(fx))
        if c is not None:
            # c 是 ∫dy/g(y) 里 ln|y| 的系数: c·ln|y| = F(x) + C ⟹ ln|y| = F/c，
            #   故通解 y = C·e^(F/c)。用 1/c 的精确形态构造指数, 避免 e^(x/0.333333) 浮点噪声。
            if abs(c - 1.0) < 1e-12:
                expo = _sym_simplify(F)
                cdisp = ""
            else:
                inv = 1.0 / c
                # 用倒数(精确形态)构造指数, 避免出现 e^(x/0.333333) 这种浮点噪声
                expo = _sym_simplify(("mul", ("num", inv), F))
                if abs(inv - round(inv)) < 1e-9:
                    cdisp = " × 1/{}".format(int(round(inv)))
                else:
                    cdisp = " × {}".format(_sym_numstr(c))
            mid = "| 4 | 左侧得 (1/c)·ln|y|{0}，右侧得 {1} |\n\n".format(cdisp, Fstr)
            return head + mid + "> 通解：**y = C * e^({0})**".format(_sym_str(expo))
        G = None
        if gy[0] == "pow" and gy[1][0] == "var" and gy[1][1] == "y" and gy[2][0] == "num":
            p = gy[2][1]
            if abs(p - 1.0) > 1e-12:
                G = ("div", ("pow", ("var", "y"), ("num", 1.0 - p)), ("num", 1.0 - p))
        if G is None:
            return None
        Gs = _sym_simplify(G)
        mid = "| 4 | 左面积分得 {0}，右面积分得 {1} + C |\n\n".format(_sym_str(Gs), Fstr)
        return head + mid + "> 隐式通解：**{0} = {1} + C**".format(_sym_str(Gs), Fstr)

    def _calc_route(self, raw):
        if not isinstance(raw, str):
            return None
        if not re.search(r"(?:求导|导数|微分|积分|∫|原函数|dy\s*/\s*dx|d\s*/\s*dx|y\s*['′])", raw, re.I):
            return None
        for fn in (self._calc_ode, self._calc_derivative, self._calc_integral):
            try:
                r = fn(raw)
            except Exception:
                r = None
            if r:
                return r
        return None

    def answer(self, raw):
        if not self.detect(raw):
            return None
        # v1.7 Alpha: 微积分(求导/积分/微分方程)抢先处理 —— 微分方程右端也含 "=",
        #   若不抢先, 会被下面的"变量方程"分支误当普通代数方程解出错误答案。
        calc = self._calc_route(raw)
        if calc:
            return calc
        norm = self._normalize(raw)
        # v1.5 正式版: 矩阵运算优先(含 [[ ]] 或 矩阵/行列式 关键词)
        if "[" in raw or any(w in raw for w in ["矩阵", "行列式", "行列"]):
            mres = self._mat_compute(raw)
            if mres:
                return mres
        # v1.5 正式版: 多元一次方程组 (高斯消元)
        # v1.7 Alpha: 放宽触发条件 —— 旧版只认 "方程组/元一次" 字样, 于是
        #   "解方程 2x+3y=12, x-y=1" 这种没写"方程组"的多元题会漏掉, 一路掉到联网搜索。
        #   现在: 只要出现 ≥2 条含字母的等式, 或等式里凑齐 ≥2 个未知数, 就交给高斯消元。
        _multi_eq = False
        if re.search(r"[=＝]", raw):
            _n_letter_eq, _lvars = 0, set()
            for _lh, _rh in re.findall(r"([0-9a-zA-Z+\-*/·×\s.]+)=([0-9a-zA-Z+\-*/·×\s.]+)", raw):
                if re.search(r"[a-df-zA-DF-Z]", _lh + _rh):
                    _n_letter_eq += 1
                    _lvars |= set(re.findall(r"[a-df-zA-DF-Z]", _lh + _rh))
            _multi_eq = (_n_letter_eq >= 2) or (len(_lvars) >= 2)
        if (any(w in raw for w in ["方程组", "元一次"]) or _multi_eq) and re.search(r"[=＝]", raw):
            lres = self._solve_linear(raw)
            if lres:
                return lres
        # v1.5 Alpha: 含"变量+等号"的方程必须先走精确求解, 否则通用数学 AST 会误把
        #   "x + 3 = 5" 抠出个 "3" 答成纯数字 → 变成"计算 3", 牛头不对马嘴。
        eq_raw = self._extract_eq(norm) or self._extract_eq(raw)
        if eq_raw:
            eq_norm = self._normalize(eq_raw)
            if "=" in eq_norm and re.search(r"[a-df-zA-DF-Z]", eq_norm):
                _eq = self._solve_equation(eq_norm)
                if _eq:
                    return _eq
        # v1.5 Alpha: 再试数学 AST(函数/根号/常量/幂) —— 纯数值的广面数学
        ast_ans = self._ast_compute(norm)
        if ast_ans:
            return ast_ans
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
# v1.6 超大型更新: 代码生成器
#   ① 需求拆解: 动手前先讲清"要什么/干什么用/怎么算完成"
#   ② 绝不空壳: 兜底也是完整可跑的真实现, 拒绝空 function
#   ③ 七语言语法表 + 代码示例库(在数据文件里), 让小方真正"学过语法"
# ============================================================
class CodeResponder:
    # 语言识别: 元组按"更长/更具体"在前排列, 避免 "c++" 被裸 "c" 抢、或 "javascript" 被 "js" 误拆。
    _LANGS = [
        ("javascript", "JavaScript"), ("java脚本", "JavaScript"), ("js脚本", "JavaScript"),
        ("node.js", "JavaScript"), ("node", "JavaScript"),
        ("js", "JavaScript"), ("typescript", "TypeScript"), ("ts", "TypeScript"),
        ("python3", "Python"), ("python", "Python"), ("py", "Python"),
        ("java", "Java"), ("kotlin", "Kotlin"),
        ("c#", "C#"), ("csharp", "C#"), ("c sharp", "C#"),
        ("c/c++", "C++"), ("c++", "C++"), ("cpp", "C++"),
        ("c语言", "C"), ("纯c", "C"), ("ansi c", "C"),
        ("golang", "Go"), ("go语言", "Go"), (" go ", "Go"),
        ("rust", "Rust"), ("php", "PHP"), ("swift", "Swift"), ("ruby", "Ruby"),
        ("lua", "Lua"), ("r语言", "R"), ("matlab", "MATLAB"),
        ("sql", "SQL"), ("bash", "Shell"), ("shell", "Shell"),
    ]
    # v1.7 Alpha: "泛化写码" 的触发词 —— 旧版只认"写代码/写程序", 于是
    #   "用 JavaScript 写一个防抖函数" 这种自然说法判不出代码意图, 一路返回 None。
    _GEN_KWS = ["写代码", "写个代码", "写程序", "写个程序", "帮我写", "帮我实现", "写一段",
                "写一个", "写个", "写一下", "实现一个", "实现个", "编写", "脚本", "代码", "编程", "实现"]
    # v1.6: Markdown 代码块的语言标注 (```python 这样)
    _BLOCK_TAG = {"Python": "python", "JavaScript": "javascript", "TypeScript": "typescript",
                  "Java": "java", "C": "c", "C++": "cpp", "C#": "csharp", "Go": "go",
                  "Rust": "rust", "PHP": "php", "Swift": "swift", "Ruby": "ruby",
                  "Kotlin": "kotlin", "Lua": "lua", "R": "r", "MATLAB": "matlab",
                  "SQL": "sql", "Shell": "bash"}

    # 每个任务: kws 命中算法意图; code = {语言: 该语言实现}; lang-note 提示备了哪几门
    _TASKS = [
        {"kws": ["冒泡", "bubble"], "title": "冒泡排序", "langs": "Python/JS/C",
         "code": {
            "Python": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        for j in range(n - 1 - i):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
            "JavaScript": "function bubbleSort(arr) {\n  for (let i = 0; i < arr.length - 1; i++) {\n    for (let j = 0; j < arr.length - 1 - i; j++) {\n      if (arr[j] > arr[j + 1]) {\n        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];\n      }\n    }\n  }\n  return arr;\n}",
            "C": "void bubbleSort(int a[], int n) {\n    int i, j, t;\n    for (i = 0; i < n - 1; i++)\n        for (j = 0; j < n - 1 - i; j++)\n            if (a[j] > a[j + 1]) { t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }\n}"},
         "exp": "思路：外层每轮把最大的数“冒泡”到末尾；内层两两比较，前大后小就交换。每轮少比较一次，N 个数共 O(N²)。"},
        {"kws": ["快速排序", "快排", "quick", "归并"], "title": "快速排序", "langs": "Python/JS/C",
         "code": {
            "Python": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    p = arr[0]\n    left = [x for x in arr[1:] if x < p]\n    right = [x for x in arr[1:] if x >= p]\n    return quick_sort(left) + [p] + quick_sort(right)",
            "JavaScript": "function quickSort(arr) {\n  if (arr.length <= 1) return arr;\n  const p = arr[0];\n  const left = arr.slice(1).filter(x => x < p);\n  const right = arr.slice(1).filter(x => x >= p);\n  return [...quickSort(left), p, ...quickSort(right)];\n}",
            "C": "void quickSort(int a[], int lo, int hi) {\n    if (lo >= hi) return;\n    int p = a[lo], i = lo, j = hi, t;\n    while (i < j) {\n        while (i < j && a[j] >= p) j--;\n        while (i < j && a[i] <= p) i++;\n        if (i < j) { t = a[i]; a[i] = a[j]; a[j] = t; }\n    }\n    a[lo] = a[i]; a[i] = p;\n    quickSort(a, lo, i - 1);\n    quickSort(a, i + 1, hi);\n}"},
         "exp": "思路：分治。任选基准 p，把比 p 小的放左边、大的放右边，再递归对两边排序。平均 O(N log N)。"},
        {"kws": ["二分", "binary"], "title": "二分查找", "langs": "Python/JS/C",
         "code": {
            "Python": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
            "JavaScript": "function binarySearch(arr, target) {\n  let lo = 0, hi = arr.length - 1;\n  while (lo <= hi) {\n    const mid = (lo + hi) >> 1;\n    if (arr[mid] === target) return mid;\n    else if (arr[mid] < target) lo = mid + 1;\n    else hi = mid - 1;\n  }\n  return -1;\n}",
            "C": "int binarySearch(int a[], int n, int target) {\n    int lo = 0, hi = n - 1, mid;\n    while (lo <= hi) {\n        mid = (lo + hi) / 2;\n        if (a[mid] == target) return mid;\n        else if (a[mid] < target) lo = mid + 1;\n        else hi = mid - 1;\n    }\n    return -1;\n}"},
         "exp": "思路：对有序数组每次砍掉一半，比较中点和目标决定去左半还是右半，复杂度 O(log N)。"},
        {"kws": ["阶乘", "factorial"], "title": "阶乘", "langs": "Python/JS",
         "code": {
            "Python": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "JavaScript": "function factorial(n) {\n  if (n <= 1) return 1;\n  return n * factorial(n - 1);\n}"},
         "exp": "思路：递归。factorial(n) = n × factorial(n-1)，边界是 n≤1 返回 1。"},
        {"kws": ["斐波那契", "fibonacci"], "title": "斐波那契", "langs": "Python/JS/C",
         "code": {
            "Python": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            "JavaScript": "function fib(n) {\n  let a = 0, b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}",
            "C": "int fib(int n) {\n    int a = 0, b = 1, i, t;\n    for (i = 0; i < n; i++) { t = b; b = a + b; a = t; }\n    return a;\n}"},
         "exp": "思路：动态规划递推，每一项是前两项之和；用双变量滚动省内存，O(N)。"},
        {"kws": ["素数", "质数", "prime"], "title": "素数判断", "langs": "Python/JS/C",
         "code": {
            "Python": "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
            "JavaScript": "function isPrime(n) {\n  if (n < 2) return false;\n  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;\n  return true;\n}",
            "C": "int isPrime(int n) {\n    int i;\n    if (n < 2) return 0;\n    for (i = 2; i * i <= n; i++)\n        if (n % i == 0) return 0;\n    return 1;\n}"},
         "exp": "思路：只需枚举到 √n，只要有一个能整除就不是素数，O(√n)。"},
        {"kws": ["最大公约数", "gcd", "欧几里得"], "title": "最大公约数", "langs": "Python/JS/C",
         "code": {
            "Python": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
            "JavaScript": "function gcd(a, b) {\n  while (b) [a, b] = [b, a % b];\n  return a;\n}",
            "C": "int gcd(int a, int b) {\n    int t;\n    while (b) { t = a % b; a = b; b = t; }\n    return a;\n}"},
         "exp": "思路：欧几里得算法，gcd(a,b)=gcd(b, a mod b)，反复取余直到余数为 0。"},
        {"kws": ["求和函数", "求和", "数组求和", "sum", "累加", "平均值", "平均数"],
         "title": "数组求和", "langs": "Python/JS",
         "code": {
            "Python": "def array_sum(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total",
            "JavaScript": "function arraySum(arr) {\n  let total = 0;\n  for (const x of arr) total += x;\n  return total;\n}"},
         "exp": "思路：初始化累加和，遍历每一项加进去。注意它和排序是两码事——排序要重新排列，求和只要累加。"},
        {"kws": ["摄氏度", "华氏度", "温度", "温度转换"], "title": "温度转换", "langs": "Python",
         "code": {
            "Python": "def c_to_f(c):\n    return c * 9 / 5 + 32\n\ndef f_to_c(f):\n    return (f - 32) * 5 / 9"},
         "exp": "思路：摄氏转华氏 ℉=℃×9/5+32；反向用 (℉-32)×5/9。"},
        {"kws": ["二进制", "进制转换", "进制"], "title": "十进制转二进制", "langs": "Python/JS",
         "code": {
            "Python": "def to_bin(n):\n    \"\"\"手写版: 反复除 2 取余, 余数倒排\"\"\"\n    if n == 0:\n        return \"0\"\n    neg = n < 0\n    n = abs(n)\n    digits = []\n    while n > 0:\n        digits.append(str(n % 2))   # 取当前最低位\n        n //= 2                     # 右移一位\n    s = \"\".join(reversed(digits))   # 余数倒排\n    return \"-\" + s if neg else s\n\ndef to_bin_builtin(n):\n    return bin(n)[2:]               # 现成函数: bin(10) -> '0b1010'\n\nprint(to_bin(10), to_bin(255), to_bin_builtin(10))   # 1010 11111111 1010",
            "JavaScript": "function toBin(n) {\n  if (n === 0) return \"0\";\n  const neg = n < 0;\n  n = Math.abs(n);\n  let s = \"\";\n  while (n > 0) {\n    s = (n % 2) + s;   // 余数往前面拼\n    n = Math.floor(n / 2);\n  }\n  return neg ? \"-\" + s : s;\n}\nconsole.log(toBin(10), (10).toString(2));   // 1010 1010"},
         "exp": "思路：反复除 2 取余，余数倒排即得二进制。手写一遍能看懂进制本质，实际写项目直接用 Python 的 bin() / JS 的 toString(2) 更省事。"},
        # —— v1.2 加强: 基础语法教程 (修复"写循环教程"被答成 Python 创始人) ——
        {"kws": ["循环", "for循环", "for 循环", "while循环", "while 循环", "遍历"],
         "title": "循环教程 (for / while)", "langs": "Python/JS",
         "code": {
            "Python": "# for: 依次取序列(range/列表/字符串)里每一个\nfor i in range(5):\n    print(i)      # 依次打印 0 1 2 3 4\n\nmy_list = [10, 20, 30]\nfor item in my_list:\n    print(item)   # 依次 10 20 30\n\n# while: 条件成立就一直执行, 记得让条件会变化\ncount = 0\nwhile count < 3:\n    print(count)\n    count += 1    # 少了这行会死循环\n\n# break 提前跳出, continue 直接进入下一次\nfor i in range(10):\n    if i == 3:\n        continue   # 跳过 3\n    if i == 6:\n        break      # 到 6 就停\n    print(i)",
            "JavaScript": "// for: 遍历\nfor (let i = 0; i < 5; i++) {\n  console.log(i);          // 0 1 2 3 4\n}\n\nconst arr = [10, 20, 30];\nfor (const item of arr) {  // for...of 取值\n  console.log(item);       // 10 20 30\n}\n\n// while: 条件成立就一直执行\nlet count = 0;\nwhile (count < 3) {\n  console.log(count);\n  count++;                 // 别忘了让条件变化, 否则死循环\n}\n\n// break 跳出整层, continue 进下一次\nfor (let i = 0; i < 10; i++) {\n  if (i === 3) continue;   // 跳过 3\n  if (i === 6) break;      // 到 6 就停\n  console.log(i);\n}"},
         "exp": "思路：for 用来遍历序列，自动取下一个元素；while 靠条件决定，只要条件为真就执行，务必让条件会变化否则死循环。break 跳出整层循环、continue 跳过本次进入下一次。想进阶就循环里写循环（嵌套），更优雅。"},
        {"kws": ["python基础", "python 基础", "python入门", "python 入门", "python教程", "python 教程",
                 "学python", "学 python", "python语法", "python 语法", "基础教程", "入门教程", "语法教程"],
         "title": "Python 基础入门教程", "langs": "Python",
         "code": {
            "Python": "# 1) 变量: 不需要声明类型, 直接赋值\nx = 5\nname = \"小方\"\n\n# 2) 条件: if / elif / else\nif x > 3:\n    print(\"x 大于 3\")\nelif x == 3:\n    print(\"正好 3\")\nelse:\n    print(\"x 小于 3\")\n\n# 3) 循环: for 遍历, while 按条件\nfor i in range(3):\n    print(i)\n\n# 4) 函数: def 定义, 调用了才执行\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n\n# 5) 容器: 列表[] / 字典{} / 元组()\nnums = [1, 2, 3, 4]\nprint(nums[0], nums[-1])\nd = {\"name\": \"小方\", \"age\": 5}\nprint(d[\"name\"])\n\n# 6) 注释: # 开头的行不会被运行"},
         "exp": "思路：Python 从 变量→条件(if/elif/else)→循环(for/while)→函数(def)→容器(列表/字典/元组) 层层递进。先照着逐行跑、改数字看输出变化，把每种语法跑通即算入门。"},
        # —— v1.6 新增任务: 更多高频算法与练习题, 全部完整实现, 绝不空壳 ——
        {"kws": ["选择排序", "select sort"], "title": "选择排序", "langs": "Python/JS/C++",
         "code": {
            "Python": "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n - 1):\n        m = i                          # 假设当前位置最小\n        for j in range(i + 1, n):\n            if arr[j] < arr[m]:\n                m = j                  # 记下更小的位置\n        arr[i], arr[m] = arr[m], arr[i]\n    return arr",
            "JavaScript": "function selectionSort(arr) {\n  for (let i = 0; i < arr.length - 1; i++) {\n    let m = i;\n    for (let j = i + 1; j < arr.length; j++) {\n      if (arr[j] < arr[m]) m = j;\n    }\n    [arr[i], arr[m]] = [arr[m], arr[i]];\n  }\n  return arr;\n}",
            "C++": "#include <vector>\n#include <algorithm>\n\nvoid selectionSort(std::vector<int>& a) {\n    for (int i = 0; i < (int)a.size() - 1; i++) {\n        int m = i;\n        for (int j = i + 1; j < (int)a.size(); j++)\n            if (a[j] < a[m]) m = j;\n        std::swap(a[i], a[m]);\n    }\n}"},
         "exp": "思路：每轮从未排序区挑最小的，放到已排序区末尾。比较 O(N²)、交换只有 O(N) 次，适合交换代价大的场景。"},
        {"kws": ["插入排序", "insert sort"], "title": "插入排序", "langs": "Python/JS",
         "code": {
            "Python": "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]        # 比它大的往后挪\n            j -= 1\n        arr[j + 1] = key               # 空位放 key\n    return arr",
            "JavaScript": "function insertionSort(arr) {\n  for (let i = 1; i < arr.length; i++) {\n    const key = arr[i];\n    let j = i - 1;\n    while (j >= 0 && arr[j] > key) {\n      arr[j + 1] = arr[j];\n      j--;\n    }\n    arr[j + 1] = key;\n  }\n  return arr;\n}"},
         "exp": "思路：像整理扑克牌，把新牌从右往左比较、插到合适位置。基本有序时接近 O(N)，是最快的简单排序之一。"},
        {"kws": ["反转字符串", "字符串反转", "反转数组", "reverse"], "title": "反转", "langs": "Python/JS/C++",
         "code": {
            "Python": "def reverse_str(s):\n    return s[::-1]               # 切片一步到位\n\ndef reverse_arr(arr):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        arr[left], arr[right] = arr[right], arr[left]\n        left += 1\n        right -= 1\n    return arr",
            "JavaScript": "function reverseStr(s) {\n  return s.split(\"\").reverse().join(\"\");\n}\n\nfunction reverseArr(arr) {\n  let left = 0, right = arr.length - 1;\n  while (left < right) {\n    [arr[left], arr[right]] = [arr[right], arr[left]];\n    left++; right--;\n  }\n  return arr;\n}",
            "C++": "#include <string>\n#include <algorithm>\n\nstd::string reverseStr(std::string s) {\n    std::reverse(s.begin(), s.end());\n    return s;\n}"},
         "exp": "思路：Python/JS 有现成切片或 reverse；手写版用双指针两头往中间换，原地完成 O(N)。"},
        {"kws": ["回文", "palindrome"], "title": "回文判断", "langs": "Python/JS",
         "code": {
            "Python": "def is_palindrome(s):\n    s = \"\".join(c.lower() for c in s if c.isalnum())   # 只留字母数字\n    return s == s[::-1]",
            "JavaScript": "function isPalindrome(s) {\n  s = s.toLowerCase().replace(/[^a-z0-9\\u4e00-\\u9fff]/g, \"\");\n  let l = 0, r = s.length - 1;\n  while (l < r) {\n    if (s[l] !== s[r]) return false;\n    l++; r--;\n  }\n  return true;\n}"},
         "exp": "思路：先清洗（去标点、统一小写），再判断正读反读是否一致。双指针写法空间 O(1)。"},
        {"kws": ["去重", "重复元素", "duplicate"], "title": "列表去重", "langs": "Python/JS",
         "code": {
            "Python": "def dedup(arr):\n    seen = set()\n    out = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)      # 保持原顺序\n    return out\n# 一行版: list(dict.fromkeys(arr))",
            "JavaScript": "function dedup(arr) {\n  return [...new Set(arr)];    // 保序去重一行版\n}\n\nfunction dedupManual(arr) {\n  const seen = new Set(), out = [];\n  for (const x of arr) {\n    if (!seen.has(x)) { seen.add(x); out.push(x); }\n  }\n  return out;\n}"},
         "exp": "思路：用集合记录见过没有，O(N) 一遍扫完；Set 版快且保序，dict.fromkeys 是 Python 一行版。"},
        {"kws": ["最大值", "最小值", "第二大", "找最值"], "title": "找最值", "langs": "Python/JS",
         "code": {
            "Python": "def find_max2(arr):\n    if len(arr) < 2:\n        return None\n    best = second = float(\"-inf\")\n    for x in arr:\n        if x > best:\n            second, best = best, x   # 旧最大降级为第二\n        elif x > second and x != best:\n            second = x\n    return best, second",
            "JavaScript": "function findMax2(arr) {\n  let best = -Infinity, second = -Infinity;\n  for (const x of arr) {\n    if (x > best) { second = best; best = x; }\n    else if (x > second && x !== best) second = x;\n  }\n  return [best, second];\n}"},
         "exp": "思路：一遍扫描维护两个变量，遇到更大的就把旧冠军降级为亚军，O(N) 同时拿到最大和第二大。"},
        {"kws": ["九九乘法表", "乘法表", "99乘法"], "title": "九九乘法表", "langs": "Python/JS/C++",
         "code": {
            "Python": "for i in range(1, 10):\n    for j in range(1, i + 1):\n        print(\"{}x{}={:<4}\".format(j, i, i * j), end='')\n    print()",
            "JavaScript": "for (let i = 1; i <= 9; i++) {\n  let line = \"\";\n  for (let j = 1; j <= i; j++) line += `${j}x${i}=${i * j}\\t`;\n  console.log(line);\n}",
            "C++": "#include <iostream>\n#include <iomanip>\n\nint main() {\n    for (int i = 1; i <= 9; i++) {\n        for (int j = 1; j <= i; j++)\n            std::cout << j << 'x' << i << '=' << std::left << std::setw(4) << i * j;\n        std::cout << '\\n';\n    }\n    return 0;\n}"},
         "exp": "思路：外层控制行(1~9)，内层只到当前行号(i)，形成三角形；print 用 end='' 防换行，行末统一换。"},
        {"kws": ["水仙花数", "narcissistic"], "title": "水仙花数", "langs": "Python/C++",
         "code": {
            "Python": "for n in range(100, 1000):\n    a, b, c = n // 100, n // 10 % 10, n % 10\n    if a**3 + b**3 + c**3 == n:\n        print(n)   # 153 370 371 407",
            "C++": "#include <iostream>\n\nint main() {\n    for (int n = 100; n < 1000; n++) {\n        int a = n / 100, b = n / 10 % 10, c = n % 10;\n        if (a*a*a + b*b*b + c*c*c == n) std::cout << n << '\\n';\n    }\n    return 0;\n}"},
         "exp": "思路：三位数各位立方和等于它自己。用整除和取余拆出百十个位即可，不必转字符串。"},
        {"kws": ["括号匹配", "有效括号", "括号", "栈"], "title": "括号匹配（栈）", "langs": "Python/JS",
         "code": {
            "Python": "def match(s):\n    pair = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n    stack = []\n    for c in s:\n        if c in \"([{\":\n            stack.append(c)             # 左括号入栈\n        elif c in pair:\n            if not stack or stack.pop() != pair[c]:\n                return False            # 右括号找不到配对\n    return not stack                    # 栈空才算全配对",
            "JavaScript": "function match(s) {\n  const pair = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"};\n  const st = [];\n  for (const c of s) {\n    if (c === \"(\" || c === \"[\" || c === \"{\") st.push(c);\n    else if (pair[c]) {\n      if (st.pop() !== pair[c]) return false;\n    }\n  }\n  return st.length === 0;\n}"},
         "exp": "思路：栈是配对问题的标准结构——左括号进栈，右括号来时看栈顶是否刚好配对，最后栈必须空。"},
        {"kws": ["反转链表", "链表反转", "链表"], "title": "反转链表", "langs": "Python/JS",
         "code": {
            "Python": "class Node:\n    def __init__(self, val, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n    prev = None\n    cur = head\n    while cur:\n        nxt = cur.next      # 先存下后面的\n        cur.next = prev     # 把指针掉头\n        prev = cur          # prev 前进\n        cur = nxt\n    return prev             # 新的头",
            "JavaScript": "function reverseList(head) {\n  let prev = null, cur = head;\n  while (cur) {\n    const nxt = cur.next;\n    cur.next = prev;\n    prev = cur;\n    cur = nxt;\n  }\n  return prev;\n}"},
         "exp": "思路：三个指针 prev/cur/nxt，逐个把 next 指向前一个。链表题先画图再写，指针顺序错一步全乱。"},
        {"kws": ["猜数字", "小游戏代码", "游戏代码"], "title": "猜数字小游戏", "langs": "Python",
         "code": {
            "Python": "import random\n\nanswer = random.randint(1, 100)\ntries = 0\nprint(\"我想好了一个 1~100 的数，猜猜看！\")\nwhile True:\n    try:\n        g = int(input(\"你猜: \"))\n    except ValueError:\n        print(\"请输入整数\")\n        continue\n    tries += 1\n    if g < answer:\n        print(\"小了\")\n    elif g > answer:\n        print(\"大了\")\n    else:\n        print(\"答对了! 共猜了\", tries, \"次\")\n        break"},
         "exp": "思路：random 生成答案，循环里读输入并给「大了/小了」反馈，猜中 break。try/except 防止输入不是数字时崩溃。"},
        {"kws": ["bmi", "体重"], "title": "BMI 计算器", "langs": "Python/JS",
         "code": {
            "Python": "def bmi(weight, height):\n    v = weight / height ** 2              # kg / m²\n    if v < 18.5:    level = \"偏瘦\"\n    elif v < 24:    level = \"正常\"\n    elif v < 28:    level = \"偏胖\"\n    else:           level = \"肥胖\"\n    return round(v, 1), level",
            "JavaScript": "function bmi(weight, height) {\n  const v = weight / height ** 2;\n  const level = v < 18.5 ? \"偏瘦\" : v < 24 ? \"正常\" : v < 28 ? \"偏胖\" : \"肥胖\";\n  return [v.toFixed(1), level];\n}"},
         "exp": "思路：公式 BMI=体重(kg)/身高²(m²)，再用 if 链分级。这种「计算+分级」结构同样适用于成绩、税率等问题。"},
        {"kws": ["面向对象", "定义一个类", "写个类", "写一个类", "类和对象", "构造函数", "oop"], "title": "类与对象入门", "langs": "Python/JS/Java",
         "code": {
            "Python": "class Student:\n    def __init__(self, name, score):\n        self.name = name        # 属性\n        self.score = score\n\n    def level(self):            # 方法\n        return \"及格\" if self.score >= 60 else \"不及格\"\n\ns = Student(\"小方\", 92)\nprint(s.name, s.level())",
            "JavaScript": "class Student {\n  constructor(name, score) {\n    this.name = name;\n    this.score = score;\n  }\n  level() {\n    return this.score >= 60 ? \"及格\" : \"不及格\";\n  }\n}\nconst s = new Student(\"小方\", 92);\nconsole.log(s.name, s.level());",
            "Java": "public class Main {\n    static class Student {\n        String name; int score;\n        Student(String name, int score) { this.name = name; this.score = score; }\n        String level() { return score >= 60 ? \"及格\" : \"不及格\"; }\n    }\n    public static void main(String[] args) {\n        Student s = new Student(\"小方\", 92);\n        System.out.println(s.name + \" \" + s.level());\n    }\n}"},
         "exp": "思路：类是模板，对象是按模板造出的实例。__init__/constructor 是构造函数，self/this 指对象自己，方法里都能访问属性。"},
        {"kws": ["异常处理", "try", "报错", "错误处理", "except", "exception"], "title": "异常处理", "langs": "Python/JS",
         "code": {
            "Python": "try:\n    n = int(input(\"数字: \"))\n    print(10 / n)\nexcept ValueError:\n    print(\"不是数字\")\nexcept ZeroDivisionError:\n    print(\"不能除以 0\")\nelse:\n    print(\"没出错才走这\")\nfinally:\n    print(\"无论如何都执行(收尾)\")",
            "JavaScript": "try {\n  const n = Number(prompt(\"数字:\"));\n  if (Number.isNaN(n)) throw new TypeError(\"不是数字\");\n  console.log(10 / n);\n} catch (e) {\n  console.log(\"出错:\", e.message);\n} finally {\n  console.log(\"无论如何都执行\");\n}"},
         "exp": "思路：try 放可能出错的代码，except/catch 接住指定错误避免崩溃，finally 收尾（关文件、释放资源）。"},
        {"kws": ["读文件", "写文件", "文件读写", "文件操作"], "title": "文件读写", "langs": "Python",
         "code": {
            "Python": "# 写: with 会自动关文件, 不用手动 close\nwith open(\"demo.txt\", \"w\", encoding=\"utf-8\") as f:\n    f.write(\"第一行\\n\")\n    f.writelines([\"第二行\\n\", \"第三行\\n\"])\n\n# 读: 逐行处理\nwith open(\"demo.txt\", \"r\", encoding=\"utf-8\") as f:\n    for i, line in enumerate(f, 1):\n        print(i, line.strip())\n\n# 一次性读全部\nwith open(\"demo.txt\", encoding=\"utf-8\") as f:\n    text = f.read()"},
         "exp": "思路：with open(...) as f 是标准姿势，出异常也保证关文件。「r」读「w」写(覆盖)「a」追加，中文文件记得 encoding 用 utf-8。"},
        {"kws": ["字典", "哈希", "hashmap", "键值对"], "title": "字典/HashMap 用法", "langs": "Python/JS",
         "code": {
            "Python": "ages = {\"小方\": 5, \"小明\": 12}\nages[\"小红\"] = 11               # 增\nages[\"小方\"] = 6                # 改\nprint(ages.get(\"小李\", 0))       # 查(键不存在给默认, 不报错)\nfor name, age in ages.items():  # 遍历\n    print(name, age)\nprint(\"小李\" in ages)           # 判断键存在",
            "JavaScript": "const ages = { \"小方\": 5, \"小明\": 12 };\nages[\"小红\"] = 11;                    // 增\nages[\"小方\"] = 6;                     // 改\nconsole.log(ages[\"小李\"] ?? 0);       // 查(空值合并给默认)\nfor (const [name, age] of Object.entries(ages)) console.log(name, age);\nconsole.log(\"小李\" in ages);          // 判断键存在"},
         "exp": "思路：字典是「键→值」映射，查找 O(1)。get(..., 默认)/?? 能避免键不存在时报错，那是新手最常见的崩溃点。"},
        # v1.7 Alpha 新增: 防抖/节流 —— 前端高频痛点, 以前问它会被判成"泛化写码"甚至返回空
        {"kws": ["防抖", "debounce", "节流", "throttle", "搜索框联想"],
         "title": "防抖 / 节流", "langs": "JS/Python",
         "code": {
            "JavaScript": "// 防抖: 停止触发 wait 毫秒后才执行一次(搜索框联想/窗口 resize)\nfunction debounce(fn, wait = 300) {\n  let timer = null;\n  return function (...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), wait);\n  };\n}\n\n// 节流: 每 wait 毫秒最多执行一次(滚动监听/按钮防连点)\nfunction throttle(fn, wait = 300) {\n  let last = 0;\n  return function (...args) {\n    const now = Date.now();\n    if (now - last >= wait) {\n      last = now;\n      fn.apply(this, args);\n    }\n  };\n}\n\n// 用法\nconst onInput = debounce(e => console.log(\"请求接口:\", e.target.value), 300);\ndocument.querySelector(\"#kw\").addEventListener(\"input\", onInput);",
            "Python": "import threading\nimport time\n\n\ndef debounce(wait=0.3):\n    \"\"\"装饰器: 连续调用时只保留最后一次(延迟 wait 秒执行)\"\"\"\n    def deco(fn):\n        state = {\"timer\": None}\n        def wrapped(*a, **k):\n            if state[\"timer\"]:\n                state[\"timer\"].cancel()\n            t = threading.Timer(wait, lambda: fn(*a, **k))\n            state[\"timer\"] = t\n            t.start()\n        return wrapped\n    return deco\n\n\n@debounce(0.3)\ndef search(kw):\n    print(\"请求接口:\", kw)\n\n\n# 连打 5 次, 只有最后一次真正发请求\nfor ch in \"小方工作室\":\n    search(ch)\ntime.sleep(0.5)"},
         "exp": "思路：**防抖**=事件停了你再动（连点 10 次只跑 1 次，等最后一次）；**节流**=不管你点多快，我固定节奏跑（每 300ms 至多 1 次）。搜索框联想、窗口 resize 用防抖；滚动加载、按钮防连点用节流。核心都是「记一个上次时间/定时器」。",},
    ]
    # v1.6: 兜底不再是空骨架, 而是"每种语言一段完整可跑的真实现"
    # (根治用户吐槽的"写个算法中间是空的, 只写了个 function")
    _GENERIC_FULL = {
        "Python": (
            "def process(data):\n"
            "    \"\"\"传入一个列表: 筛选→加工→汇总 的完整小例子\"\"\"\n"
            "    picked = [x for x in data if x > 0]      # 1) 筛选: 只要正数\n"
            "    changed = [x * 2 for x in picked]        # 2) 加工: 全部翻倍\n"
            "    return sum(changed) if changed else 0    # 3) 汇总: 求和返回\n"
            "\n"
            "print(process([3, -1, 4, -1, 5]))    # 输出 24"
        ),
        "JavaScript": (
            "function process(data) {\n"
            "  const picked = data.filter(x => x > 0);      // 1) 筛选\n"
            "  const changed = picked.map(x => x * 2);      // 2) 加工\n"
            "  return changed.reduce((s, x) => s + x, 0);   // 3) 汇总\n"
            "}\n"
            "console.log(process([3, -1, 4, -1, 5]));  // 输出 24"
        ),
        "Java": (
            "import java.util.*;\n\n"
            "public class Main {\n"
            "    static int process(int[] data) {\n"
            "        int total = 0;\n"
            "        for (int x : data) {              // 1) 遍历\n"
            "            if (x > 0) total += x * 2;    // 2) 筛选+加工\n"
            "        }\n"
            "        return total;                     // 3) 汇总\n"
            "    }\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(process(new int[]{3, -1, 4, -1, 5})); // 24\n"
            "    }\n"
            "}"
        ),
        "C": (
            "#include <stdio.h>\n\n"
            "int process(int a[], int n) {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < n; i++)            /* 遍历 */\n"
            "        if (a[i] > 0) total += a[i] * 2;   /* 筛选+加工 */\n"
            "    return total;                          /* 汇总 */\n"
            "}\n\n"
            "int main(void) {\n"
            "    int a[] = {3, -1, 4, -1, 5};\n"
            "    printf(\"%d\\n\", process(a, 5));   /* 输出 24 */\n"
            "    return 0;\n"
            "}"
        ),
        "C++": (
            "#include <iostream>\n#include <vector>\n\n"
            "int process(const std::vector<int>& v) {\n"
            "    int total = 0;\n"
            "    for (int x : v)\n"
            "        if (x > 0) total += x * 2;   // 筛选+加工\n"
            "    return total;                    // 汇总\n"
            "}\n\n"
            "int main() {\n"
            "    std::cout << process({3, -1, 4, -1, 5}) << \"\\n\";  // 输出 24\n"
            "    return 0;\n"
            "}"
        ),
        "C#": (
            "using System;\n\nclass Program {\n"
            "    static int Process(int[] data) {\n"
            "        int total = 0;\n"
            "        foreach (int x in data)\n"
            "            if (x > 0) total += x * 2;   // 筛选+加工\n"
            "        return total;                    // 汇总\n"
            "    }\n"
            "    static void Main() {\n"
            "        Console.WriteLine(Process(new[] { 3, -1, 4, -1, 5 })); // 24\n"
            "    }\n"
            "}"
        ),
        "Go": (
            "package main\n\nimport \"fmt\"\n\n"
            "func process(data []int) int {\n"
            "    total := 0\n"
            "    for _, x := range data {\n"
            "        if x > 0 { total += x * 2 }  // 筛选+加工\n"
            "    }\n"
            "    return total                     // 汇总\n"
            "}\n\n"
            "func main() {\n"
            "    fmt.Println(process([]int{3, -1, 4, -1, 5})) // 24\n"
            "}"
        ),
        # —— v1.6: 补齐到 18 门, "用 X 写代码" 不再回落 Python ——
        "TypeScript": (
            "function process(data: number[]): number {\n"
            "  const picked = data.filter((x: number) => x > 0);        // 1) 筛选\n"
            "  const changed = picked.map((x: number) => x * 2);        // 2) 加工\n"
            "  return changed.reduce((s: number, x: number) => s + x, 0); // 3) 汇总\n"
            "}\n"
            "console.log(process([3, -1, 4, -1, 5]));   // 输出 24"
        ),
        "Kotlin": (
            "fun process(data: List<Int>): Int {\n"
            "    var total = 0\n"
            "    for (x in data) {                  // 1) 遍历\n"
            "        if (x > 0) total += x * 2      // 2) 筛选 + 加工\n"
            "    }\n"
            "    return total                       // 3) 汇总\n"
            "}\n\n"
            "fun main() {\n"
            "    println(process(listOf(3, -1, 4, -1, 5)))   // 输出 24\n"
            "}"
        ),
        "Rust": (
            "fn process(data: &[i32]) -> i32 {\n"
            "    data.iter()\n"
            "        .filter(|&&x| x > 0)      // 1) 筛选\n"
            "        .map(|&x| x * 2)          // 2) 加工\n"
            "        .sum()                    // 3) 汇总\n"
            "}\n\n"
            "fn main() {\n"
            "    println!(\"{}\", process(&[3, -1, 4, -1, 5]));   // 输出 24\n"
            "}"
        ),
        "PHP": (
            "<?php\n"
            "function process(array $data): int {\n"
            "    $total = 0;\n"
            "    foreach ($data as $x) {            // 1) 遍历\n"
            "        if ($x > 0) $total += $x * 2;  // 2) 筛选 + 加工\n"
            "    }\n"
            "    return $total;                     // 3) 汇总\n"
            "}\n"
            "echo process([3, -1, 4, -1, 5]), PHP_EOL;   // 输出 24"
        ),
        "Swift": (
            "func process(_ data: [Int]) -> Int {\n"
            "    return data.filter { $0 > 0 }   // 1) 筛选\n"
            "               .map { $0 * 2 }      // 2) 加工\n"
            "               .reduce(0, +)        // 3) 汇总\n"
            "}\n"
            "print(process([3, -1, 4, -1, 5]))   // 输出 24"
        ),
        "Ruby": (
            "def process(data)\n"
            "  data.select { |x| x > 0 }   # 1) 筛选\n"
            "      .map { |x| x * 2 }      # 2) 加工\n"
            "      .sum                    # 3) 汇总\n"
            "end\n"
            "puts process([3, -1, 4, -1, 5])   # 输出 24"
        ),
        "Lua": (
            "function process(data)\n"
            "    local total = 0\n"
            "    for _, x in ipairs(data) do                 -- 1) 遍历\n"
            "        if x > 0 then total = total + x * 2 end -- 2) 筛选 + 加工\n"
            "    end\n"
            "    return total                                -- 3) 汇总\n"
            "end\n"
            "print(process({3, -1, 4, -1, 5}))               -- 输出 24"
        ),
        "R": (
            "process <- function(data) {\n"
            "  picked  <- data[data > 0]   # 1) 筛选\n"
            "  changed <- picked * 2       # 2) 加工\n"
            "  sum(changed)                # 3) 汇总\n"
            "}\n"
            "cat(process(c(3, -1, 4, -1, 5)), \"\\n\")   # 输出 24"
        ),
        "MATLAB": (
            "function total = process(data)\n"
            "    picked  = data(data > 0);   % 1) 筛选\n"
            "    changed = picked * 2;       % 2) 加工\n"
            "    total   = sum(changed);     % 3) 汇总\n"
            "end\n"
            "% 调用: disp(process([3, -1, 4, -1, 5]))   % 输出 24"
        ),
        "SQL": (
            "-- 场景: 统计订单表里\"正数金额翻倍后的总和\"\n"
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount INTEGER);\n"
            "INSERT INTO orders (amount) VALUES (3), (-1), (4), (-1), (5);\n\n"
            "SELECT SUM(amount * 2) AS total   -- 2) 加工 + 3) 汇总\n"
            "FROM orders\n"
            "WHERE amount > 0;                 -- 1) 筛选\n"
            "-- 输出 total = 24"
        ),
        "Shell": (
            "#!/bin/bash\n"
            "# 1) 定义数据  2) 筛选+加工  3) 汇总\n"
            "total=0\n"
            "for x in 3 -1 4 -1 5; do\n"
            "    if [ \"$x\" -gt 0 ]; then\n"
            "        total=$((total + x * 2))\n"
            "    fi\n"
            "done\n"
            "echo \"$total\"   # 输出 24"
        ),
    }
    _GENERIC_EXP = ("思路：写代码三件事——① 定义数据；② 用条件(if)和循环(for/while)处理批量；③ 返回结果。"
                    "上面是一段完整可跑的{lang}示例（筛选→加工→汇总三步），照着改成你的数据就能跑。"
                    "把【用哪门语言】+【具体算法/要解决什么问题】+【输入输出长啥样】告诉我，我直接给完整成品，绝不空壳。")

    def __init__(self):
        self.detected = []

    def _all_tasks(self):
        """内置任务 + 数据文件里的代码示例库(小方"学过"的示例也参与匹配)"""
        return self._TASKS + list(getattr(DATA, "CODE_EXAMPLES", []) or [])

    def _detect_lang(self, raw):
        s = " " + (raw or "").lower().strip() + " "
        for kw, name in self._LANGS:
            if kw in s:
                return name
        return None

    def _best_task(self, low):
        """命中多个任务时, 取'命中的关键词最长'的那个 → 泛词(排序/循环)不抢具体词(归并排序/循环队列)"""
        best, best_len = None, 0
        for t in self._all_tasks():
            for k in t.get("kws", []):
                k = (k or "").lower()
                if k and k in low and len(k) > best_len:
                    best, best_len = t, len(k)
        return best

    def detect(self, raw):
        low = (raw or "").lower()
        if self._best_task(low):
            return True
        if any(k in low for k in self._GEN_KWS):
            return True
        lang = self._detect_lang(raw)
        if lang and any(k in low for k in ["语法", "速查", "怎么学", "怎么写"]):
            return True
        return False

    def _pick(self, t, lang):
        cm = t["code"]
        if lang and lang in cm:
            return lang, cm[lang], ""
        main = None
        for cand in ("Python", "JavaScript", "C++", "C", "Java"):
            if cand in cm:
                main = cand
                break
        if main is None:
            main = next(iter(cm))
        if lang:
            note = "（我这里现成的是 {} 版，先用它给你参考，要 {} 原版我随时补）".format(
                "/".join(cm.keys()), lang)
        else:
            note = ""
        # v1.6 修正: 走"退而用现成版"这条路时, 标注的语言必须是"真正给出的那份代码"的语言,
        #   否则会出现「说是 Rust 完整实现, 里面却是 Python 代码」的错配（代码块标签也跟着错）。
        return main, cm[main], note

    def _syntax_sheet(self, lang):
        """七语言语法表(来自数据文件 CODE_SYNTAX); 没有就返回 None"""
        sheet = (getattr(DATA, "CODE_SYNTAX", {}) or {}).get(lang)
        if sheet:
            return "📚 **{0} 语法速查**：\n```{1}\n{2}\n```".format(
                lang, self._BLOCK_TAG.get(lang, ""), sheet)
        return None

    def _decompose(self, raw, lang):
        """需求拆解: 用 Markdown 讲清'要什么/干什么用/怎么算完成'"""
        what = raw if raw and len(raw) <= 30 else (raw[:30] + "…")
        return ("📋 **需求拆解**\n"
                "- **要什么**：{0}\n"
                "- **干什么用**：跑通一段完整程序逻辑（输入 → 处理 → 输出）\n"
                "- **怎么算完成**：{1} 完整可运行实现，带测试输出，绝不空壳").format(what, lang)

    def reply(self, raw):
        low = (raw or "").lower()
        lang = self._detect_lang(raw)
        # ⓪ v1.6: "X 语法速查" 是明确要语法表, 优先喂语法表(不被内置教程抢)
        if lang and any(k in low for k in ["语法", "速查", "语法表", "语法速查"]):
            sheet = self._syntax_sheet(lang)
            if sheet:
                return sheet + "\n想深入哪块语法（循环/函数/类/文件…）说一声，我给专项讲解+完整示例。"
        # ① 明确任务 → 完整实现 + 思路讲解 (取命中最具体的那条)
        t = self._best_task(low)
        if t:
            lang2, code, note = self._pick(t, lang)
            head = "💻 **{0}**（{1} 完整实现）".format(t["title"], lang2)
            tag = self._BLOCK_TAG.get(lang2, "")
            return "{0}\n```{3}\n{1}\n```\n{2}{4}".format(head, code, note, tag, t["exp"])
        # ② 语法请求 → 喂语法表
        if lang and any(k in low for k in ["语法", "速查", "怎么学", "怎么写"]):
            sheet = self._syntax_sheet(lang)
            if sheet:
                return sheet + "\n想深入哪块语法（循环/函数/类/文件…）说一声，我给专项讲解+完整示例。"
        # ③ 泛化写码 → 需求拆解 + 该语言完整可跑示例 + 追问三要素
        if any(k in low for k in self._GEN_KWS):
            name = lang or "Python"
            if name in self._GENERIC_FULL:
                shown, note = name, ""
            else:
                shown = "Python"
                note = "（{0} 完整示例我马上补，先给 Python 版参考）".format(name)
            head = self._decompose(raw, shown)
            return ("{0}\n\n💻 **{1} 完整可跑示例**：\n```{2}\n{3}\n```\n{4}{5}".format(
                head, shown, self._BLOCK_TAG.get(shown, ""),
                self._GENERIC_FULL[shown], note,
                self._GENERIC_EXP.format(lang=shown)))
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
    MEM_FILE = "xiaofang_memory_v17alpha.json"
    _LEGACY_FILES = ("xiaofang_memory_v16.json",
                     "xiaofang_memory_v15.json",
                     "xiaofang_memory_v15alpha.json",
                     "xiaofang_memory_v14alpha.json",
                     "xiaofang_memory_v13alpha.json",
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
            for legacy in self._LEGACY_FILES:            # 旧版本记忆文件 → 迁移到 v16
                lp = self._mem_path(legacy)
                if not os.path.exists(lp):
                    # v1.6: 旧版记忆随版本一起归档在 历史版本\<ver>\ 下, 子目录里也找一遍, 别丢老底子
                    import glob as _glob
                    hits = _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "历史版本", "*", legacy))
                    if hits:
                        lp = hits[0]
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
            data = {"version": VERSION, "seq": self._seq,
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
        self.lm.vocab.add(TERMINATOR)   # v1.7: 终止符进词表 → 生成能"输出直到终止符"再收尾
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
        # v1.6 检索修复: 联网先判"库里有没有", 联网后要"进网页取正文"
        self.last_web = None
        self.last_web_reason = ""
        self.last_web_backend = ""
        # v1.7 Alpha: 多轮投稿 —— 记住上一版作文/代码, 用户说「再改改 / 换算法 / 加长」时接着改
        self.last_essay = None
        self.last_code = None
        self.last_kind = None
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
            _is_code = self._is_code_query(user_input)
            _bk_desc = ("开发者平台加权: " + " → ".join(CODE_SEARCH_SITES)) if _is_code \
                else "→".join(SEARCH_BACKENDS)
            parts.append("  库里没现成的, 判定需联网 → 关键词:「{}」 后端: {}".format(wq, _bk_desc))
            wres = self._fetch_web(wq, code=_is_code)
            self.last_web = {"query": wq, "results": wres, "ok": bool(wres), "time": time.time()}
            if wres:
                parts.append("  命中 {} 条, 开始进网页取正文:".format(len(wres)))
                for i, item in enumerate(wres[:4], 1):
                    parts.append("     {}「{}」".format(i, (item.get("title") or "")[:30]))
                # v1.6 检索修复 ③: 不只吃摘要 —— 真打开前几条网页, 把正文的字抠出来
                wres = self._deep_read(wres, want=2, max_chars=1400)
                self.last_web["results"] = wres
                _got = sum(1 for r in wres if r.get("page"))
                if _got:
                    parts.append("  → 已进入 {} 个网页抠出正文(各约 {} 字), 注入上下文做整合".format(
                        _got, len(wres[0].get("page", "")) or 0))
                else:
                    parts.append("  → 网页正文抓取受限, 退回用搜索摘要整合")
            else:
                parts.append("  → 检索未命中(网络受限), 回退本地生成")
        elif getattr(self, "last_web_reason", "") == "db_hit":
            parts.append("【②+ 联网判定】")
            parts.append("  → 知识库已有高置信对齐记忆, 就地取材, 不联网")
        else:
            parts.append("【②+ 联网判定】")
            parts.append("  → 库内足以作答(reason={}), 不联网".format(getattr(self, "last_web_reason", "") or "db_enough"))
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
        # v1.7 Alpha: 把"权重不再固定"这件事显性化 —— 优化器 + 反向传播的实况
        _tb = self._train_brief()
        if _tb:
            parts.append("【④+ 优化器与反向传播 (v1.7 可训练权重)】")
            parts.append("  AdamW(lr={}, β=(0.9,0.999), wd=0.01) · 训练层=最后{}层FFN + 输出门控/残差适配器"
                         .format(TRAIN_LR, TRAIN_LAST_BLOCKS))
            parts.append("  " + _tb)
            parts.append("  近步 loss 曲线: " + self._train_hist_line())
        # ── v1.7 Alpha 核心: 两层嵌套深度思考, 全部由 Transformer 本体逐词生成, 每层各验算一次 ──
        #   第①层想(嵌入→多头自注意力→FFN→logits→softmax→top-p 采样, 撞上 </s> 收尾)
        #     → 验算①(意图贴合度 cos / 分布置信度 / 注意力落点)
        #     → 把①层思考 + 纠偏意见再喂回同一个 Transformer
        #     → 第②层想 → 验算② → 双验通过才真正成文。
        t1 = self._tf_think(user_input, bias_extra=(kb_top or ""), n_tok=NEST_TOK)
        v1 = self._tf_verify(t1, intent, 1)
        _s2 = ((t1.get("text") or "") + " " + (v1.get("hint") or "")).strip() or user_input
        t2 = self._tf_think(_s2, bias_extra=(kb_top or ""), n_tok=NEST_TOK)
        v2 = self._tf_verify(t2, intent, 2)
        self.last_nested = {"t1": t1, "v1": v1, "t2": t2, "v2": v2}
        parts.append("【⑤ 两层嵌套深度思考 (Transformer 本体生成)】")
        parts.append("  第①层想(逐词采样, 收在终止符 {}): 「{}」".format(
            TERMINATOR, (t1.get("text") or "—")[:70]))
        parts.append(self._nested_attn_line(t1, 1))
        parts.append("  " + v1["line"])
        parts.append("  第②层想(①层思考+验算纠偏再喂回同一 Transformer): 「{}」".format(
            (t2.get("text") or "—")[:70]))
        parts.append(self._nested_attn_line(t2, 2))
        parts.append("  " + v2["line"])
        parts.append("  双验定稿: " + ("两层都通过 ✓ 直接成文" if (v1["ok"] and v2["ok"])
                                     else "第②层已按纠偏重想 → 收敛成文"))
        parts.append("【⑥ 结论】")
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
        mono = self._nested_monologue(user_input, t1, v1, t2, v2, intent, web_should,
                                      emo, kb_hits, kb_top, kb_score, detail, strategies)
        tail = ["  ⏱ 本次思考已耗时 {:.2f} 秒 (含深度思考打字显示)".format(elapsed), "</deep_think>"]
        if SHOW_DEEP_THINK:
            for _ln in mono:                       # 内心独白逐字打字 → 更像在"边想边敲"
                _typewrite(_ln, C_DEEP, delay=0.02)
            _typewrite(C_HINT + "────── 底层神经运算痕迹(供技术查验) ──────", C_DEEP)
            _typewrite_lines(parts, C_DEEP)        # 算法过程保留
            _typewrite_lines(tail, C_DEEP)
        return "\n".join(mono + parts + tail)

    # ══════════════════════════════════════════════════════════════════════
    # v1.7 Alpha · 两层嵌套深度思考 (全部由 Transformer 本体生成 + 每层一次验算)
    #   第①层思考: 嵌入 → 多头自注意力 → FFN → logits → softmax → top-p 采样,
    #               逐词生成(不是模板); 遇到终止符 </s> 收尾。
    #   验算①: 用同一个 Transformer 给这段思考算【意图贴合度(余弦) + 分布置信度(归一化熵)
    #           + 注意力峰值落在哪个词】, 不合格就把"纠偏提示"反馈进第二层。
    #   第②层思考: 把第①层思考 + 验算结论 + 检索知识一起送回 Transformer 再生成一遍。
    #   验算②: 再验一次, 双验通过才真正给答案 —— 这就是"嵌套两层"。
    # ══════════════════════════════════════════════════════════════════════
    def _tf_think(self, seed_text, bias_extra="", n_tok=26, temp=0.88):
        """v1.7 Alpha: 由 Transformer 本体逐词生成的思考片段(嵌入→注意力→FFN→softmax→采样)。
        返回 dict: text / top1 / cos / conf / entropy / attn_peak / attn_tok / ids。"""
        tr = self.transformer
        seed = self.tokenizer.tokenize(seed_text or "")[:SEED_TOKENS] or ["好"]
        bias = set(seed) | set(self.tokenizer.tokenize(bias_extra or ""))
        try:
            outline = tr.generate(seed, bias, max_tokens=max(4, int(n_tok)),
                                  temperature=float(temp), top_p=0.92)
        except Exception:
            outline = list(seed)
        tail = [t for t in outline[len(seed):] if t != TERMINATOR]   # 终止符不写进正文
        text = "".join(tail)
        ids = [tr.token2id.get(t, 0) for t in (seed + tail)][:TRAIN_MAX_SEQ] or [0]
        top1 = cos = ent = conf = 0.0
        attn_peak, attn_tok = -1, ""
        try:
            probs, trc = tr.forward(ids, trace=True)
            V = max(2, len(probs))
            top1 = float(min(1.0, max(0.0, np.max(probs))))
            ent = float(-np.sum(probs * np.log(probs + 1e-12)))
            conf = float(min(1.0, max(0.0, 1.0 - ent / np.log(V))))   # 归一化熵 → 置信度(钳到 [0,1])
            # 意图贴合度: 生成内容与用户输入的词嵌入中心余弦(同一嵌入空间, 越贴题越接近 1)
            E = np.asarray(_to_host(tr.embed.val()), dtype=np.float64)
            sid = [tr.token2id.get(t, 0) for t in seed] or [0]
            gid = [tr.token2id.get(t, 0) for t in tail]
            ps = E[sid].mean(axis=0)
            pg = E[gid].mean(axis=0) if gid else ps
            cos = float(min(1.0, max(-1.0, np.dot(pg, ps) / ((np.linalg.norm(pg) * np.linalg.norm(ps)) + 1e-9))))
            ar = trc.get("attn_head0_last")
            if ar is not None:
                ar = np.asarray(_to_host(ar), dtype=np.float64).reshape(-1)
                if ar.size:
                    p = int(np.argmax(ar))
                    attn_peak = p
                    toks = seed + tail
                    if 0 <= p < len(toks):
                        attn_tok = str(toks[p])
        except Exception:
            pass
        return {"text": text, "top1": top1, "cos": cos, "conf": conf, "entropy": ent,
                "attn_peak": attn_peak, "attn_tok": attn_tok, "ids": ids}

    def _tf_verify(self, thought, intent, layer=1):
        """v1.7 Alpha: 验算层 —— 用同一套 Transformer 指标判定该层思考是否"想对了"。
        返回 dict: ok / cos / conf / attn_tok / hint(给下一层的纠偏提示) / line(可读结论)。"""
        cos = float(thought.get("cos", 0.0))
        conf = float(thought.get("conf", 0.0))
        ok = (cos >= 0.18) and (conf >= 0.10)
        why = ""
        if cos < 0.18:
            why = "生成内容与用户输入的中心语义夹角偏大(有点跑题)"
        elif conf < 0.10:
            why = "输出分布过于发散、没有咬定重点"
        hint = "" if ok else ("聚焦用户真正问的意图 " + str(intent.get("top", "")))
        line = ("验算{}: 意图贴合度 cos={:.3f} · 分布置信度={:.3f} · 注意力峰值落点「{}」 → {}"
                .format(layer, cos, conf, thought.get("attn_tok") or "—",
                        "通过 ✓" if ok else ("需再想一遍 ✗（{}）".format(why))))
        return {"ok": ok, "cos": cos, "conf": conf,
                "attn_tok": thought.get("attn_tok", ""), "hint": hint, "line": line}

    def _nested_attn_line(self, t, layer):
        """v1.7: 把该层思考的真实注意力画面摊开(哪个头、盯住哪个词、分布多集中)。"""
        return ("  第{}层注意力: {} 头 · 第0头末位 query 最强盯住「{}」(位置 {}) · top1={:.4f} · 归一化熵={:.3f}"
                .format(layer, self.transformer.n_heads, t.get("attn_tok") or "—",
                        t.get("attn_peak", -1), float(t.get("top1", 0.0)), float(t.get("entropy", 0.0))))

    def _nested_monologue(self, user_input, t1, v1, t2, v2, intent, web_should,
                          emo, kb_hits, kb_top, kb_score, detail, strategies):
        """v1.7 Alpha: 内心独白不再套模板 —— 正文就是 Transformer 本体①/②层逐词采样出来的字,
        这里只加"串场"的人话外壳; 万一两层都没采出成句, 才退回旧的意图独白兜底(不让用户看到空白)。"""
        def _txt(d):
            s = (d.get("text") or "").strip()
            return s if s else "…（这层没采出成句，我按验算意见接着往下想）"
        if not (t1.get("text") or t2.get("text")):
            return self._think_monologue(user_input, emo, intent, kb_hits, kb_top, kb_score,
                                         detail, strategies, web_should)
        _u = (user_input or "").strip().replace("\n", " ")[:30]
        out = ["唔，你说的是「{}」，我先把词元喂进 {} 头注意力的神经矩阵里过一遍。".format(
            _u, self.transformer.n_heads),
            "第①层（我自己逐词想的）: {}".format(_txt(t1)),
            "  " + v1["line"]]
        if not v1["ok"]:
            out.append("  验算没过，我把「{}」这条纠偏意见连同①层思考再次喂回 Transformer。".format(v1["hint"] or "聚焦真正的问题"))
        out.append("第②层（吃进①层思考 + 验算结论再想）: {}".format(_txt(t2)))
        out.append("  " + v2["line"])
        if web_should:
            out.append("  库里没现成答案，我已经联网把实时内容取回来，一起揉进答复。")
        out.append("  " + ("两验都过，思路立住了，按这个往下成文。" if v2["ok"]
                           else "第二层还有发散，我按验算意见收一收再给答案。"))
        return out

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
            _pg = sum(1 for _r in self.last_web.get("results", []) if _r.get("page"))
            mono.append("本地库里没现成的，我上网查了「{}」，{}条结果，还进了{}个网页把正文读了读。".format(
                _wq, _wn, _pg))
        elif getattr(self, "last_web_reason", "") == "db_hit":
            mono.append("这个我脑子里刚好有：「{}」，拿来直接用，不用上网折腾。".format(kb_hits[0][1]["t"]))
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

    # v1.7 Alpha: 问"小方自己/作者/用什么做的/什么框架" → 本地答, 绝不联网(不曲解)
    _SELF_ASK = ["是什么框架", "什么框架", "用什么做", "用什么写", "用什么开发", "什么语言写",
                 "用什么写得", "用的什么", "用什么模型", "什么模型",
                 "作者是谁", "谁做的", "谁开发", "谁创造",
                 "谁发明", "谁创建", "谁设计", "怎么做的", "怎么做出来",
                 "怎么写的", "技术栈", "底层", "源码",
                 "小方工作室", "工作室", "fanggame"]
    # v1.7: 只有"带自我指向(你/您/小方/咱)"的才算在问小方本人,
    #   否则像"Vue 是什么框架"这种正常技术提问会被误判成自家问题而不给联网。
    _SELF_REF = ["你", "您", "小方", "咱", "fanggame"]

    def _is_self_ask(self, text):
        t = (text or "")
        if any(m in t for m in ["小方工作室", "工作室", "fanggame"]):
            return True
        if not any(m in t for m in self._SELF_REF):
            return False
        return any(m in t for m in self._SELF_ASK)

    def _needs_online(self, text, intent=None, kb_score=0.0, has_question=False):
        # v1.1 正式版: 深度理解"到底要不要上网搜" —— 不滥用知识库, 也不乱联网。
        # 返回 True = 该上网搜; False = 用本地/直接作答。
        t = (text or "").strip()
        if not t:
            return False
        # ⚠ 凡是问"小方自己/作者/用什么做的/什么框架/小方工作室" → 本地答, 绝不联网(不曲解)
        if self._is_self_ask(t):
            return False
        # v1.7 Alpha: 代码类诉求 → 自动联网, 且检索加权 GitHub/Gitee/CSDN 等开发者平台
        if self._is_code_query(t):
            return True
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
        # v1.7 Alpha: 自家/工作室情报只走本地精准靶向 —— 网上根本查不到, 联网只会答歪,
        #   所以这里直接短路, 不参与后面的"低置信概念题自动联网"判定。
        if self._is_self_ask(text):
            self.last_web_reason = "self_local"
            return False
        kb_score = kb_hits[0][0] if kb_hits else 0.0
        # ---- v1.6 检索修复 ①: 数据库优先 ----
        # 明确的实时诉求(让查/最新/天气/股价…) → 一定要联网, 不受库命中影响
        hard_web = bool(intent and intent.get("top") in ("search", "weather")) or any(
            m in text for m in ["搜一下", "查一下", "帮我查", "上网查", "查查", "搜搜", "上网",
                                "去搜", "最新", "实时", "今天", "现在", "股价", "天气", "汇率",
                                "比分", "多少钱", "多少钱"])
        # 库里已有高置信且主题对齐的记忆 → 就地取材, 不联网
        if (not hard_web) and kb_hits and kb_score >= 0.5 and _kb_aligned(text, kb_hits[0][1]):
            self.last_web_reason = "db_hit"
            return False
        # v1.1 正式版: 先用"该联网"决策 (含推荐/攻略/自我排除)
        if self._needs_online(text, intent, kb_score, emo and emo.get("has_question")):
            self.last_web_reason = "explicit"
            return True
        if any(m in text for m in ["不知道", "不了解", "不清楚", "不懂", "查一下", "帮我查"]):
            self.last_web_reason = "unknown"
            return True
        # 自我意图理解: 问"是什么/怎么/如何/介绍" 且本地知识库置信低 → 自动上网
        # v1.6 修正: 原先还要求 emo["has_question"], 可"什么是X？"这种问法 has_question 并不置位,
        #   结果"库里没有的概念题"全被吞成 db_enough、不上网 → 违背"库里没有就上网"。改看问句特征词本身。
        # v1.7 Alpha: 全局自动联网 —— 概念题之外, 再把"谁/哪年/什么时候/多少/哪里/为什么"这类
        #   事实型提问一并纳入; 用户不必说"搜一下", 该查就自己查。
        _concept_q = any(m in text for m in ["是什么", "什么是", "什么意思", "怎么样", "怎么",
                                             "如何", "怎样", "介绍", "原理", "起源", "区别", "在哪",
                                             "谁", "哪年", "哪一年", "什么时候", "多少", "多少钱",
                                             "哪里", "为什么", "为何", "哪一个", "哪个", "排名",
                                             "现状", "进展", "怎么样", "介绍下", "介绍一下"])
        if intent["top"] in ("question", "study", "start", "search") and kb_score < 0.42 and _concept_q:
            self.last_web_reason = "low_conf"
            return True
        self.last_web_reason = "db_enough"
        return False

    def _extract_query(self, text):
        # 由用户输入生成联网关键词
        q = re.sub(r"[？?。！!，,、；;\s]+", " ", text)
        for w in ["小方", "帮我", "请", "搜索", "搜一下", "查一下", "上网查",
                  "查找", "是什么", "什么是", "呢", "啊", "帮我查一下"]:
            q = q.replace(w, "")
        q = q.strip(" :：")
        return q[:40] or (text.strip()[:40] or "AI")

    def _baidu_search(self, query, max_results=5):
        """v1.4: 百度搜索(国内稳定可用)。返回 [{title, body, href}, ...]，与 ddgs 格式兼容。"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        url = "https://www.baidu.com/s"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            resp = requests.get(url, params={"wd": query, "rn": max_results * 2},
                                headers=headers, timeout=8)
            resp.encoding = "utf-8"
        except Exception:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for c in soup.select("div.result, div.c-container, div[class*='result']"):
            title_tag = c.select_one("h3 a, h3.title a, .t a")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            href = title_tag.get("href", "")
            body = ""
            for sel in [".c-abstract", ".content-right_8Zs40", "span.content-right_8Zs40",
                        ".c-span-last", "div[class*='abstract']", "div[class*='content-right']"]:
                tag = c.select_one(sel)
                if tag:
                    body = tag.get_text(strip=True)
                    break
            if not body:
                all_text = c.get_text(" ", strip=True)
                body = all_text[len(title):].strip()[:200] if all_text else ""
            if title and body:
                results.append({"title": title, "body": body, "href": href})
            if len(results) >= max_results:
                break
        return results

    def _bing_search(self, query, max_results=5):
        """v1.6: 必应(cn.bing.com)搜索 —— 返回真实直达链接(不是跳转广告页),
        这样才能"进网页看正文"。返回 [{title, body, href}, ...]。"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            resp = requests.get("https://cn.bing.com/search",
                                params={"q": query, "ensearch": 0},
                                headers=headers, timeout=8)
            if resp.status_code != 200:
                return []
            resp.encoding = resp.apparent_encoding or "utf-8"
            html = resp.text or ""
        except Exception:
            return []
        soup = BeautifulSoup(html, "lxml")
        results = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a:
                continue
            title = a.get_text(strip=True)
            href = a.get("href", "")
            p = li.select_one("p")
            body = p.get_text(" ", strip=True) if p else ""
            if not body:
                body = li.get_text(" ", strip=True)[len(title):].strip()[:200]
            if title and href.startswith("http"):
                results.append({"title": title, "body": body, "href": href})
            if len(results) >= max_results:
                break
        return results

    # v1.7 Alpha: 代码类检索加权 —— 收到写代码/报错/API 这类诉求时,
    #   不能一味只翻百度必应, 要优先命中开发者平台(GitHub / Gitee / CSDN / Stack Overflow / GitCode)。
    _CODE_HINT = [
        "代码", "程序", "函数", "脚本", "报错", "错误", "异常", "bug", "debug", "调试",
        "编译", "运行", "安装", "依赖", "环境", "接口", "api", "sdk", "库", "框架",
        "前端", "后端", "爬虫", "数据库", "sql", "正则", "算法", "排序", "递归", "指针",
        "类", "继承", "多线程", "并发", "异步", "部署", "服务器", "git", "docker",
        "npm", "pip", "conda", "python", "java", "javascript", "typescript", "c++",
        "cpp", "c#", "golang", "rust", "php", "swift", "kotlin", "html", "css",
        "vue", "react", "node", "pandas", "numpy", "pytorch", "tensorflow", "opencv",
        "matplotlib", "爬取", "traceback", "exception", "syntaxerror", "importerror",
        "报错信息", "源码", "开源",
    ]

    def _is_code_query(self, text):
        """v1.7 Alpha: 判断是不是代码/技术类诉求 —— 是的话检索要加权开发者平台。"""
        t = (text or "").lower()
        if not t:
            return False
        return any(k in t for k in self._CODE_HINT)

    def _code_site_search(self, query, max_results=5):
        """v1.7 Alpha: 代码类检索加权 —— 依次用 site: 限定 GitHub/Gitee/CSDN 等开发者平台。
        命中即带 site 标签返回, 让"进网页看正文"落在真正的源码 / 问答 / 教程页上。"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        out, seen = [], set()
        for site in CODE_SEARCH_SITES:
            if len(out) >= max_results:
                break
            try:
                resp = requests.get("https://cn.bing.com/search",
                                    params={"q": "site:{} {}".format(site, query)},
                                    headers=headers, timeout=8)
                if resp.status_code != 200:
                    continue
                resp.encoding = resp.apparent_encoding or "utf-8"
                soup = BeautifulSoup(resp.text or "", "lxml")
                for li in soup.select("li.b_algo"):
                    a = li.select_one("h2 a")
                    if not a:
                        continue
                    href = a.get("href", "")
                    if not href.startswith("http") or href in seen:
                        continue
                    seen.add(href)
                    p = li.select_one("p")
                    out.append({"title": a.get_text(strip=True),
                                "body": p.get_text(" ", strip=True) if p else "",
                                "href": href, "site": site})
                    if len(out) >= max_results:
                        break
            except Exception:
                continue
        return out

    def _fetch_web(self, query, max_results=5, code=None):
        # v1.6 检索修复 ②: 先后端顺序 = 必应(直达链接, 能进网页取正文) → 百度 → ddgs 各后端。
        # v1.7 Alpha: 代码类查询先走开发者平台加权(GitHub/Gitee/CSDN/Stack Overflow/GitCode), 再回落通用后端。
        # v1.3 Alpha 卡死防呆: 整段网络放在守护线程硬超时里跑。
        _code = self._is_code_query(query) if code is None else code

        def _go():
            # ⓪ v1.7: 代码类 → 先打开发者平台(site: 限定), 命中就不必再翻通用搜索
            if _code:
                try:
                    cs = self._code_site_search(query, max_results=max_results)
                    if cs:
                        self.last_web_backend = "code:" + (cs[0].get("site") or "dev")
                        return cs
                except Exception:
                    pass
            # ① 先试必应: 给的是真实网址, 后续"进网页看正文"才有的可看
            try:
                bing_res = self._bing_search(query, max_results=max_results)
                if bing_res:
                    self.last_web_backend = "bing"
                    return bing_res
            except Exception:
                pass
            # ② 必应不行再试百度
            try:
                baidu_res = self._baidu_search(query, max_results=max_results)
                if baidu_res:
                    self.last_web_backend = "baidu"
                    return baidu_res
            except Exception:
                pass
            # ③ 都不行再试 ddgs 各后端
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
        # 搜索+解析需要更多时间，给 12 秒；拿不到就 [] → reply 走离线兜底，绝不卡死
        return _run_with_timeout(_go, timeout=12.0, default=[])

    # ---- v1.6 检索修复 ③: 联网不止看摘要, 要"进网页去看", 把正文的字抠出来 ----
    def _html_to_text(self, html, max_chars=1400):
        """从网页 HTML 里抽正文纯文本(去脚本/样式/导航等噪声)。纯函数, 方便单独验。"""
        if not html:
            return ""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return ""
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception:
                return ""
        for bad in soup(["script", "style", "noscript", "header", "footer", "nav",
                         "aside", "form", "iframe", "svg", "button"]):
            bad.decompose()
        # 先找语义主容器(正文区), 找不到再退回 body
        node = None
        for sel in ["article", "main", "div#content", "div.content", "div.article",
                    "div#main", "div.post", "div.entry", "div#article", "div.article-content"]:
            try:
                tag = soup.select_one(sel)
            except Exception:
                tag = None
            if tag and len(tag.get_text(" ", strip=True)) >= 80:
                node = tag
                break
        if node is None:
            node = soup.body or soup
        # 逐段收集: 太短的(导航/版权/按钮)直接丢, 只留下像正文的段;
        # 小标题本身很短但信息量大, 门槛放宽到 4 字
        chunks = []
        for p in node.find_all(["h1", "h2", "h3", "p", "li", "td"]):
            s = p.get_text(" ", strip=True)
            _low = 4 if p.name in ("h1", "h2", "h3") else 12
            if len(s) >= _low:
                chunks.append(s)
        text = " ".join(chunks) if chunks else node.get_text(" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:max_chars]

    def _fetch_page_text(self, href, max_chars=1400):
        """打开一条结果页, 抽出正文纯文本。"""
        if not href or not isinstance(href, str) or not href.startswith("http"):
            return ""
        try:
            import requests
        except ImportError:
            return ""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            resp = requests.get(href, headers=headers, timeout=6)
            if resp.status_code != 200:
                return ""
            enc = (resp.encoding or "").lower()
            if not enc or enc in ("iso-8859-1", "ascii"):
                resp.encoding = resp.apparent_encoding or "utf-8"
            html = resp.text or ""
        except Exception:
            return ""
        return self._html_to_text(html, max_chars=max_chars)

    def _deep_read(self, results, want=2, max_chars=1400):
        """v1.6: 对前 want 条结果真进网页取正文, 写回 item["page"]。
        整体套硬超时守护线程, 单页 6 秒, 拿不到就退回原摘要 —— 绝不卡死。"""
        if not results:
            return results

        def _go():
            got = 0
            for item in results:
                if got >= want:
                    break
                if item.get("page"):
                    got += 1
                    continue
                href = item.get("href") or item.get("url", "")
                if not href:
                    continue
                page = self._fetch_page_text(href, max_chars=max_chars)
                if page and len(page) >= 60:
                    item["page"] = page
                    got += 1
            return results

        try:
            return _run_with_timeout(_go, timeout=15.0, default=results) or results
        except Exception:
            return results

    def search_and_integrate(self, query, emo):
        # v0.4Search: 优先复用思考内已检索的结果, 否则立即联网; RAG 式整合
        web = getattr(self, "last_web", None)
        if web and web.get("ok") and time.time() - web.get("time", 0) < 30:
            wq, results = web["query"], web["results"]
        else:
            wq = self._extract_query(query)
            results = self._fetch_web(wq, code=self._is_code_query(query))
            self.last_web = {"query": wq, "results": results, "ok": bool(results), "time": time.time()}
        if not results:
            return "这次没能联网拿到结果(网络或后端受限)。换个关键词试试？或者我可以基于本地知识来回答。"
        # v1.6 检索修复 ③: 光看摘要不够 —— 真进网页把正文抠出来, 再谈整合
        if not any(r.get("page") for r in results):
            results = self._deep_read(results, want=2, max_chars=1400)
            if web is not None:
                web["results"] = results
            if getattr(self, "last_web", None) is not None:
                self.last_web["results"] = results
        result_texts = []
        for item in results:
            title = item.get("title", "")
            body = item.get("body", "") or ""
            page = item.get("page", "") or ""
            # 正文比摘要长就优先用正文; 否则退回摘要
            src_text = page if len(page) > len(body) else body
            result_texts.append("{}。{}".format(title, src_text))
            self.lm.add_text(title + "。" + src_text)
        # v0.5 正式版: 自升级大脑——从抓到的网页内容里学生词/生解释
        learn_extra = ""
        try:
            lw2, ld2 = self.learner.learn(" ".join(result_texts))
            if lw2 or ld2:
                learn_extra = "🧠 网页资料我已吸收：新词 {} 个，知识 {} 条。".format(lw2, ld2)
        except Exception:
            pass
        # v1.6 检索修复 ④: 把网页抠出来的字和"用户问的问题"对齐, 按相关度挑句再整合 —— 不无脑拼接
        q = (query or "").strip()
        q_bg = set()
        for _i in range(len(q) - 1):
            _bg = q[_i:_i + 2]
            if re.match(r"^[\u4e00-\u9fff]{2}$", _bg):
                q_bg.add(_bg)
        cand, seen_s = [], set()
        for rt in result_texts:
            for s in _split_sents(rt):
                s = s.strip(" ")
                if len(s) < 8:
                    continue
                key = re.sub(r"[\s\W_]+", "", s)      # 归一化去重: 标点/空白不同也算同一句
                if key in seen_s:
                    continue
                seen_s.add(key)
                hit = sum(1 for bg in q_bg if bg in s)
                cand.append((hit, s))
        cand.sort(key=lambda x: -x[0])                # 相关度高的排前, 同分保持网页原序
        picked = [s for _h, s in cand[:7]]
        # 句子之间补句读, 不能糊成一坨
        buf = []
        for s in picked:
            if buf and not buf[-1].endswith(("。", "！", "？", "…", "；", "，", "、")):
                buf.append("。")
            buf.append(s)
        gen_text = "".join(buf) if buf else result_texts[0][:80]
        gen_text = self.generator._insert_emojis(gen_text, emo)
        # 用"真抓到正文的那条"当出处, 更名副其实
        top = next((r for r in results if r.get("page")), results[0])
        src = top.get("href") or top.get("url", "")
        got_page = sum(1 for r in results if r.get("page"))
        head = "我进网页读过了" if got_page else "我从网上查到"
        answer = "{}（{}）：{}".format(head, top.get("title", "")[:24], gen_text)
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
                                "简历", "邮件", "通知", "一句话", "标题", "简介",
                                # v1.5 Alpha: 算法/代码意图绝不被作文抢走
                                "排序", "冒泡", "快排", "递归", "查找", "二分", "遍历",
                                "算法", "素数", "质数", "阶乘", "斐波那契", "求和",
                                "js", "javascript", "python", "java", "c++", "cpp",
                                "c语言", "go", "rust", "php", "swift", "typescript"]):
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
        # v1.6: 先把"字数要求 / 必须包含 / 题目自定"这类附加条款从主题里切掉, 免得主题被污染
        topic = re.split(r"(?:必须包含|必须写到|必须写|要包含|要写到|需包含|需要包含|需要写到"
                         r"|要有|要提到|必须提到|涉及|不少于|不超过|题目自定|自拟|自定"
                         r"|题目不限|题材不限|字数不限|字数要求)", topic)[0]
        topic = re.sub(r"[0-9０-９]+\s*(?:字|个字|汉字)", "", topic)
        topic = re.sub(r"[一二两三四五六七八九十百千]+\s*(?:字|个字|汉字)", "", topic)
        # 清洗主题: 先去标点, 再去句尾问词/口语词/包裹引号
        topic = topic.strip(" ，。！？,.、；;：:的'\"“”‘’「」『』【】（）()<>《》")
        for _ in range(3):
            dropped = False
            for tail in ("作文", "小作文", "文章", "短文", "感想", "心得", "论述", "主题", "内容",
                         "题目", "文字", "怎么写", "给我", "帮我", "写一篇", "写一段", "我来写",
                         "写", "呢", "啊", "吧", "呀", "了", "的", "关于"):
                if topic.endswith(tail):
                    topic = topic[:-len(tail)]
                    dropped = True
            if not dropped:
                break
            topic = topic.strip(" ，。！？,.、；;：:的'\"“”‘’「」『』【】（）()<>《》")
        topic = topic.strip(" ")
        topic = topic[:18]
        # v1.5 正式版: 允许单字主题(如"花"), 但绝不能为空或纯噪声 → 命中实体才算数
        if not topic or topic in ("这个", "那个", "一个", "一篇", "一段", "篇", "段", "个",
                                  "写", "关于", "围绕"):
            return "这个话题", "这件事"
        return topic, topic if len(topic) <= 16 else "这件事"

    # ---- v1.6: 作文靶向拆解的两个解析器 (明确字数 / 明确必须包含的内容) ----
    _CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    # 用户点名"必须包含"的要点 → 逐条落成句子时用的模板
    _MUST_TMPL = [
        "先说「{c}」——它在「{t}」里不是陪衬，而是把整篇立起来的那根柱子",
        "绕不开的还有「{c}」，它让「{t}」从一个空泛的话题，变成一段能落到地上的经历",
        "再看「{c}」，它和「{t}」是连在一起的：把「{c}」看清了，对「{t}」的理解也就深了一层",
        "值得一提的是「{c}」，正是它让「{t}」不只是道理，而有了可以触摸的温度",
        "而「{c}」这一层也不该被略过，它让「{t}」的说法有了具体的着落，不再是悬空的议论",
        "不能漏掉「{c}」，它是「{t}」这条线上最容易被忽略、却也最见分量的一环",
        "就「{c}」而言，它给「{t}」添的不是装饰，而是让人信服的那点实感",
    ]

    def _extract_len(self, text):
        """从「写800字 / 不少于600字 / 约五百字」解析目标字数; 解析不到返回 None"""
        t = text or ""
        m = re.search(r"(\d{2,5})\s*(?:字|个字|汉字)", t)
        if m:
            n = int(m.group(1))
            if 50 <= n <= 5000:
                return n
        m2 = re.search(r"([一二两三四五六七八九十百千]{1,4})\s*字", t)
        if m2:
            val, cur, unit = 0, 0, {"十": 10, "百": 100, "千": 1000}
            for ch in m2.group(1):
                if ch in self._CN_NUM:
                    cur = self._CN_NUM[ch]
                elif ch in unit:
                    val += (cur or 1) * unit[ch]
                    cur = 0
            val += cur
            if 50 <= val <= 5000:
                return val
        return None

    def _extract_must(self, text):
        """解析「必须包含/要有/要写到 X、Y」里被点名的要点清单"""
        t = text or ""
        m = re.search(r"(?:必须包含|必须写到|必须写|要包含|要写到|需包含|需要包含|需要写到"
                      r"|要有|要提到|必须提到|涉及)([^。！？；\n]{2,60})", t)
        if not m:
            return []
        seg = m.group(1)
        # v1.6: "要有A要有B / 必须包含A必须包含B" 这类重复口令也当分隔符, 免得两件要点粘成一条
        seg = re.sub(r"(?:必须包含|必须写到|必须写|要包含|要写到|需包含|需要包含|需要写到"
                     r"|要有|要提到|必须提到|涉及)", "、", seg)
        seg = re.split(r"(?:这些|等内容|之类|等等|然后|并且|同时|而且|还要|再写|写一篇|写成)", seg)[0]
        items = []
        for p in re.split(r"[、,，和及与/]|以及", seg):
            p = p.strip(" 　的了吧呢啊")
            for tail in ("等内容", "等", "内容", "作文", "文章", "短文", "文字"):
                if p.endswith(tail) and len(p) > len(tail):
                    p = p[:-len(tail)]
            p = p.strip(" 　的了吧呢啊")
            # v1.6: "必须包含A、B，800字" 里的字数不是要点, 别把 "800字" 也当成必须包含的内容
            if re.fullmatch(r"[0-9０-９]{1,5}\s*(?:字|个字|汉字)", p) or \
               re.fullmatch(r"[一二两三四五六七八九十百千]{1,5}\s*字", p):
                continue
            if 1 <= len(p) <= 14 and p not in items:
                items.append(p)
        return items[:6]

    def _gen_essay(self, text, emo, kb_hits):
        # v1.1 Alpha: 稳定长文(500~800字)。每个句子都取自完整句库, 只替换主题/事实锚点,
        #   绝不逐字随机采样 → 输出整段完整稳定的句子, 不胡言乱语。
        import random as _r
        topic, sub = self._extract_topic(text)
        t = topic or "这个话题"
        c = sub or t or "它"
        # v1.6 靶向拆解: 先明确"字数"和"必须包含的内容", 再按拆解结果成文
        target = self._extract_len(text)
        user_len = target is not None
        target = target or 600
        must = self._extract_must(text)
        # v1.6: 没给主题(或说"题目自定")但点名了要包含的内容 → 取第一个要点当主题
        if t in ("这个话题", "这件事") and must:
            t = c = must[0]
        # v1.6: 骨架随目标字数伸缩 —— 300 字不注水, 1500 字充分展开
        if target <= 400:
            n_a = n_b = n_c = 1
        elif target <= 700:
            n_a = n_b = n_c = 2
        elif target <= 1100:
            n_a = n_b = n_c = 3
        else:
            n_a = n_b = n_c = 4

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
        for _ in range(n_a):
            _a = pick(getattr(DATA, "ESSAY_BODY_A", []), used)
            if _a: parts.append(_a)
        _f = pick(getattr(DATA, "ESSAY_FACT", []), used)
        if _f: parts.append(_f)
        # 意义(n_b 句) + 应用/深化(n_c 句)
        for _ in range(n_b):
            _b = pick(getattr(DATA, "ESSAY_BODY_B", []), used)
            if _b: parts.append(_b)
        for _ in range(n_c):
            _c2 = pick(getattr(DATA, "ESSAY_BODY_C", []), used)
            if _c2: parts.append(_c2)

        # ②.5 v1.6: 用户点名"必须包含"的要点 → 逐条落成句子, 一条不漏且模板不重样
        _tmpl = list(self._MUST_TMPL)
        _r.shuffle(_tmpl)
        for _i, _m in enumerate(must):
            parts.append(_tmpl[_i % len(_tmpl)].replace("{t}", t).replace("{c}", _m))

        # ② v1.6: 按目标字数补长 —— 不到位才展开, 到位就停; 一轮不够再换序来一轮
        def _cur():
            return sum(cz(p) for p in parts)

        _need = target * 0.95
        _pool = list(getattr(DATA, "ESSAY_DEEP", []) or [])
        for _name in ("ESSAY_OPEN", "ESSAY_BODY_A", "ESSAY_BODY_B", "ESSAY_BODY_C"):
            for _s in (getattr(DATA, _name, []) or []):
                if _s not in _pool:
                    _pool.append(_s)
        _seen = set(parts)
        for _pass in range(2):
            _r.shuffle(_pool)
            for _d in _pool:
                if _cur() >= _need:
                    break
                _s = fill(_d)
                if _s in _seen:
                    continue
                _seen.add(_s)
                parts.append(_s)
            if _cur() >= _need:
                break


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
            if not p:
                continue
            if p.endswith("："):
                # v1.6: 段末只能是句读 —— 冒号悬空一律收成句号
                p = p[:-1] + "。"
            elif not p.endswith(("。", "！", "？", "～")):
                p += "。"
            clean.append(p)

        total = sum(cz(p) for p in clean)
        # ④ v1.6: Markdown 排版 —— 起/承/转/合 分 4~5 段, 段间空行, 不糊成一坨
        groups = []
        if clean:
            groups.append(clean[0])
            rest = clean[1:-1] if len(clean) > 1 else []
            if rest:
                step = max(1, (len(rest) + 2) // 3)
                for _i in range(0, len(rest), step):
                    groups.append("".join(rest[_i:_i + step]))
            if len(clean) > 1:
                groups.append(clean[-1])
        body = "\n\n".join(g for g in groups if g and g.strip())

        # ⑤ v1.6: 先给「需求拆解」小节让用户确认理解无误, 再上 Markdown 正文
        head = ["📋 **需求拆解**",
                "- **主题**：以「{0}」为题".format(t),
                "- **字数**：{0}".format(
                    "按你说的，约 {0} 字".format(target) if user_len
                    else "你没指定，我先按 {0} 字写（要改说一声）".format(target)),
                "- **必须包含**：{0}".format("、".join(must) if must else "无硬性要求，按主题正常展开"),
                "- **文体与排版**：通用文体；全角标点，段末以句号/叹号收尾，段间空行，Markdown 排版"]
        return _normalize_emoji("{0}\n\n## {1}\n\n{2}\n\n> 全文约 {3} 字".format(
            "\n".join(head), str(t), body, total))

    def _ask_options(self, question, opts, multi=False):
        """v1.5 正式版 询问机制: 选项由 AI 自定(数量/内容 AI 选), 末项恒为「其他」, 支持单选/多选。"""
        kind = "（可多选，用逗号分隔；也可以直接说你的想法）" if multi else "（单选一个，或直接说你的想法）"
        lines = ["{}. {}".format(i, o) for i, o in enumerate(opts, 1)]
        lines.append("{}. 其他[请说明]".format(len(opts) + 1))
        return "📋 {}\n{}\n{}".format(question, kind, "\n".join(lines))

    # ===============================================================
    # v1.7 Alpha: 多轮投稿 —— 作文 / 代码接着上一版迭代, 不重复输出同一份原文
    # ===============================================================
    _REVISE_KWS = ["再改改", "改改", "改一下", "再改", "换个角度", "换一个角度", "换角度",
                   "换个算法", "换算法", "加长", "写长", "长一点", "写短", "短一点",
                   "改结尾", "结尾改", "继续写", "接着写", "再来一版", "再写一版",
                   "优化一下", "再优化", "加个功能", "加功能", "润色", "扩写", "缩写",
                   "换个说法", "重写一下", "再润色", "加一段", "删掉", "去掉", "改进一下",
                   "换一种", "再深入", "深化一下", "精炼一下",
                   # v1.7 Alpha: 补自然改法(用户口语: 改成/改为/换成/加注释/迭代/递归 ...)
                   "改成", "改为", "换成", "换用", "加注释", "注释一下", "写注释", "补注释",
                   "去掉注释", "迭代版", "递归版", "非递归", "用栈", "别用递归",
                   "详细点", "详细一点", "更详细", "简单点", "简单一点", "精简", "压缩一下",
                   "分步骤", "加个例子", "加示例", "加上测试", "加测试", "加个测试",
                   "加上异常处理", "加异常处理", "加类型", "类型标注", "改成类", "封装成",
                   "换个语言", "换语言", "改用", "重写", "再重写", "收敛一下", "风格换"]
    _REV_ROUND = ["第一稿", "第二稿", "第三稿", "第四稿", "第五稿",
                  "第六稿", "第七稿", "第八稿"]

    def _is_revise(self, text):
        """v1.7: 判断这句是不是"接着上一版改"的短指令(而非新起一篇)。"""
        t = (text or "").strip()
        if not t or len(t) > 40:
            return False
        return any(k in t for k in self._REVISE_KWS)

    def _revise_last(self, text, emo, kb_hits):
        """v1.7 多轮投稿: 有上一版代码/作文时, 把"这轮的修改要求"并回去, 出下一稿。"""
        want_code = self.last_kind == "code" or (self.last_kind is None and self.last_code)
        want_essay = self.last_kind == "essay" or (self.last_kind is None and self.last_essay)
        # ① 接着上一版代码改 → 出下一版代码
        if want_code and self.last_code:
            base = self.last_code
            lang = self.code._detect_lang(text)
            _req2 = "{0} 要求：{1}".format(base.get("req", "写代码"), text)
            if lang:
                _req2 = "{0} 用{1}".format(_req2, lang)
            ca = self.code.reply(_req2)
            if ca:
                n = base.get("round", 1) + 1
                self.last_code = {"req": base.get("req", ""), "lang": lang or base.get("lang"),
                                  "text": ca, "round": n}
                self.last_kind = "code"
                return ("🔁 **多轮投稿 · {0}** 在上一版代码基础上按你说的「{1}」继续改：\n\n{2}".format(
                    self._REV_ROUND[min(n - 1, len(self._REV_ROUND) - 1)], text, ca))
        # ② 接着上一版作文改 → 把修改要求并回原主题重新成篇(句库不重样), 版本号 +1
        if want_essay and self.last_essay:
            base = self.last_essay
            n = base.get("round", 1) + 1
            topic = base.get("topic") or ""
            new_text = "写一篇关于{0}的作文 {1}".format(topic, text) if topic \
                else "写一篇作文 {0}".format(text)
            out = self._gen_essay(new_text, emo, kb_hits)
            self.last_essay = {"topic": topic, "text": out, "round": n}
            self.last_kind = "essay"
            return ("🔁 **多轮投稿 · {0}** 按「{1}」在上一版基础上重写：\n\n{2}".format(
                self._REV_ROUND[min(n - 1, len(self._REV_ROUND) - 1)], text, out))
        return None

    def _maybe_ask(self, text, intent):
        # ① 代码/算法请求但没确认语言 → 反问语言(选项 AI 定, 末项恒为"其他", 单选)
        if intent.get("top") == "code":
            has_lang = bool(self.code._detect_lang(text))
            asks_code = any(k in text for k in
                            ["写代码", "写个代码", "写程序", "写一段", "编程", "实现",
                             "函数", "排序", "递归", "算法", "脚本", "冒泡", "代码", "帮我写"])
            if asks_code and not has_lang:
                return self._ask_options(
                    "您没有确认语言，请选择您的语言",
                    ["Python", "JavaScript", "Java", "C++", "C", "Go", "C#"])
        # ② 作文请求但主题没抓住 → 反问主题
        if self._is_essay(text):
            t, _ = self._extract_topic(text)
            if t in ("这个话题", "这件事"):
                # v1.6: 虽然没给题目, 但点名了"必须包含 X、Y" → 已有可落笔的抓手, 不再反问, 直接成文
                if self._extract_must(text):
                    return None
                return self._ask_options(
                    "作文主题我拿不准，您想让我写什么？",
                    ["由我自由发挥一个主题", "写一件具体的物件/人物/场景", "先把这个主题给我，我再展开"])
        return None

    # ===============================================================
    # v1.7 Alpha: 小方工作室「精准靶向」本地路由
    #   规则: 自家资料只走本地库 (STUDIO_LOCAL_ONLY), 绝不联网 —— 网上搜不到, 搜了就是脏数据。
    #   问一件答一件: 问小说只答小说+阅读网址, 绝不夹带纪念日/口号/官网等没被问到的信息。
    # ===============================================================
    def _studio_scope(self, text):
        return ("工作室" in text) or ("fanggame" in text)

    def _studio_novel_names(self):
        return [n["name"] for n in DATA.STUDIO_NOVELS] + list(DATA.STUDIO_NOVELS_EXTRA)

    def _studio_novel_hits(self, text):
        return [n for n in self._studio_novel_names() if n in text]

    def _studio_genre_pick(self, text):
        """用户说了口味(日常/冒险/程序员) -> 按题材过滤, 其它题材主动排除。"""
        for genre, kws in DATA.NOVEL_GENRE_BUCKETS:
            if not any(k in text for k in kws):
                continue
            items = [n for n in DATA.STUDIO_NOVELS
                     if (genre in n.get("tags", [])) or (genre in n.get("genre", ""))]
            if not items:
                continue
            excluded = [n["name"] for n in DATA.STUDIO_NOVELS if n not in items]
            return genre, items, excluded
        return None

    def _studio_novels_answer(self, text, name_hits):
        novels = DATA.STUDIO_NOVELS
        # ① 点名了某一部 -> 只答那一部 (名字 + 阅读网址), 不扩散
        if name_hits:
            sel = [n for n in novels if n["name"] == name_hits[0]]
            if sel:
                n = sel[0]
                block = "**《{}》**（题材 {}，{}）\n\n{}".format(
                    n["name"], n["genre"], n["status"], n["desc"])
                block += "\n\n阅读网址：{}".format(n["url"]) if n["url"] else "\n\n（暂未上架，敬请期待）"
                return block
            return "《{}》是工作室的番外 / 支线篇目，暂未单独上架独立阅读页。".format(name_hits[0])
        # ② 点名了整个系列 (如《小方趣生活》系列) -> 只列该系列
        series = [n for n in novels if n.get("series") and n["series"] in text]
        if series:
            out = ["**《{}》系列** 目前这些：".format(series[0]["series"]), ""]
            for n in series:
                first = n["desc"].split("。")[0] + "。" if n["desc"] else ""
                out.append("- **《{}》**（{}，{}）— {}".format(n["name"], n["genre"], n["status"], first))
                out.append("  - 阅读网址：{}".format(n["url"] if n["url"] else "暂未上架，敬请期待"))
            return "\n".join(out)
        # ③ 说了口味 -> 先过滤, 只留对口的, 其它题材一句带过
        picked = self._studio_genre_pick(text)
        if picked is not None:
            genre, items, excluded = picked
            out = ["按 **{}** 向的口味，给你这几本：".format(genre), ""]
            for n in items:
                first = n["desc"].split("。")[0] + "。" if n["desc"] else ""
                out.append("- **《{}》**（{}，{}）— {}".format(n["name"], n["genre"], n["status"], first))
                out.append("  - 阅读网址：{}".format(n["url"] if n["url"] else "暂未上架，敬请期待"))
            if excluded:
                out.append("")
                out.append("（{} 是其它题材，这次就不掺进来了。）".format(
                    "、".join("《{}》".format(e) for e in excluded)))
            return "\n".join(out)
        # ③ 只是问「有哪些 / 几本」-> 全列, 只给名字 + 阅读网址
        out = ["小方工作室目前一共 **{} 部** 小说：".format(len(novels)), ""]
        for i, n in enumerate(novels, 1):
            out.append("{}. **《{}》**（{}，{}）".format(i, n["name"], n["genre"], n["status"]))
            out.append("   - 阅读网址：{}".format(n["url"] if n["url"] else "暂未上架，敬请期待"))
        out.append("")
        out.append("另有番外 / 支线篇目：{}。".format(
            "、".join("《{}》".format(e) for e in DATA.STUDIO_NOVELS_EXTRA)))
        return "\n".join(out)

    # ===============================================================
    # v1.7 Alpha: 创作类综合任务 —— 现场成篇, 本地直接产出成品
    #   诗歌 / 小游戏设计 / 小说灵感 都属于「让我给你做一个」的活儿,
    #   必须当场写出来; 绝不丢给联网搜索(搜回来是词典页那种脏结果)。
    # ===============================================================
    _POEM_MAKE = ["写诗", "写一首诗", "写首诗", "作诗", "吟诗", "赋诗", "来一首诗",
                  "来首诗", "写首", "写一首", "作一首", "来一首", "打油诗", "现代诗", "吟一首"]
    _POEM_STOP = ["有哪些", "是谁", "哪首", "赏析", "翻译", "作者", "全文", "什么意思", "背诵", "出处"]
    _GAME_MAKE = ["设计", "做", "开发", "策划", "编", "写", "来一个", "想一个",
                  "想个", "搞一个", "搭一个", "做一个", "来一款"]
    _GAME_STOP = ["推荐", "好玩", "玩什么", "排行", "评测", "好玩吗", "有哪些"]
    _IDEA_WORD = ["灵感", "构思", "创意", "点子", "构想", "大纲", "开篇", "题材", "写什么"]

    def _creative_kind(self, text):
        """判定现场创作的类别: poem / game / idea / None。"""
        t = text or ""
        stop_poem = any(s in t for s in self._POEM_STOP)
        if not stop_poem:
            if any(k in t for k in self._POEM_MAKE):
                return "poem"
            if "诗" in t and any(k in t for k in ["写", "作", "来", "给我", "帮我", "一首", "吟"]):
                return "poem"
        if any(k in t for k in ["游戏设计", "设计游戏", "设计小游戏", "设计一款游戏", "设计一个游戏"]):
            return "game"
        if ("游戏" in t) and any(k in t for k in self._GAME_MAKE) \
                and not any(s in t for s in self._GAME_STOP):
            return "game"
        if any(k in t for k in self._IDEA_WORD):
            if "灵感" in t or any(k in t for k in
                                  ["小说", "故事", "剧情", "剧本", "题材", "写作", "创作"]):
                return "idea"
        return None

    def _creative_route(self, text):
        kind = self._creative_kind(text)
        if kind == "poem":
            return self._poem_answer(text)
        if kind == "game":
            return self._game_design_answer(text)
        if kind == "idea":
            return self._novel_idea_answer(text)
        return None

    def _poem_topic(self, text):
        t = (text or "").strip()
        m = re.search(r"关于(.{1,12}?)(?:的诗|的诗歌|一首诗|诗|诗歌)", t)
        if m:
            return m.group(1).strip(" 的了")
        m = re.search(r"以(.{1,12}?)(?:为题|为主题|为话题)", t)
        if m:
            return m.group(1).strip(" 的了")
        tp, _ = self._extract_topic(t)
        tp = re.sub(r"(的诗|的诗歌|一首诗|诗|诗歌|一首)$", "", tp or "").strip(" 的了")
        if tp and tp not in ("这个话题", "这件事") and len(tp) <= 10:
            return tp
        return ""

    def _poem_answer(self, text):
        topic = self._poem_topic(text)
        best, bs = None, 0
        for p in DATA.POEM_BANK:
            s = sum(1 for kw in p.get("kws", []) if kw and (kw in text or (topic and kw in topic)))
            if s > bs:
                best, bs = p, s
        if best is not None:
            out = ["📖 **{}**".format(best.get("title", "一首小诗")),
                   "> {}".format(best.get("style", "现代诗")), ""]
            out += list(best.get("lines", []))
            if best.get("note"):
                out += ["", "—— {}".format(best["note"])]
            return "\n".join(out)
        name = topic or "此刻"
        tpl = DATA.POEM_TPL[sum(ord(c) for c in name) % len(DATA.POEM_TPL)]
        out = ["🖋 **《关于{}的一首小诗》**".format(name),
               "> {}".format(tpl.get("style", "现代诗")), ""]
        out += [ln.replace("{t}", name) for ln in tpl.get("lines", [])]
        if tpl.get("note"):
            out += ["", "—— {}".format(tpl["note"])]
        return "\n".join(out)

    def _game_design_answer(self, text):
        best, bs = None, 0
        for g in DATA.GAME_DESIGNS:
            s = sum(1 for kw in g.get("kws", []) if kw and kw in text)
            if s > bs:
                best, bs = g, s
        if best is None:
            best = DATA.GAME_DESIGNS[0]
        out = ["🎮 **{} · 游戏设计稿**".format(best.get("name", "《一分钟方块》")), "",
               "**一句话**：{}".format(best.get("tagline", "")),
               "**人数**：{}　**引擎**：{}".format(best.get("players", "单人"), best.get("engine", "tkinter")),
               "", "## 核心循环"]
        out += ["- {}".format(x) for x in best.get("loop", [])]
        if best.get("ops"):
            out += ["", "## 操作"] + ["- {}".format(x) for x in best["ops"]]
        if best.get("levels"):
            out += ["", "## 关卡与难度曲线"] + ["- {}".format(x) for x in best["levels"]]
        if best.get("art"):
            out += ["", "## 美术与音效", best["art"]]
        if best.get("tech"):
            out += ["", "## 技术要点", best["tech"]]
        if best.get("code"):
            out += ["", "## 可直接运行的 Python 原型（标准库，无需 GPU）",
                    "```python", best["code"].rstrip(), "```"]
        out += ["", "## 4 天开发计划"] + ["- {}".format(x) for x in DATA.GAME_PLAN]
        return "\n".join(out)

    def _novel_idea_answer(self, text):
        best, bs = None, 0
        for n in DATA.NOVEL_IDEAS:
            s = sum(1 for kw in n.get("kws", []) if kw and kw in text)
            if s > bs:
                best, bs = n, s
        if best is not None:
            out = ["💡 **小说灵感 · {}**  {}".format(best.get("genre", "通用"), best.get("title", "")), "",
                   "**一句话**：{}".format(best.get("logline", "")),
                   "**背景**：{}".format(best.get("setting", "")),
                   "**主角**：{}".format(best.get("hero", "")),
                   "", "## 三幕结构"] + ["- {}".format(a) for a in best.get("acts", [])]
            if best.get("twist"):
                out += ["", "**反转**：{}".format(best["twist"])]
            if best.get("hook"):
                out += ["", "**开篇试写**", best["hook"]]
            if best.get("titles"):
                out += ["", "**备选书名**：" + "、".join(best["titles"])]
            return "\n".join(out)
        name = self._idea_subject(text) or "那件说不清来路的小东西"
        title = DATA.CREATIVE_EXTRA[sum(ord(c) for c in name) % len(DATA.CREATIVE_EXTRA)]
        out = ["💡 **小说灵感 · 通用套路**  {}".format(title), "",
               "**一句话**：" + DATA.IDEA_TPL["logline"].format(name),
               "", "## 三幕结构"] + ["- {}".format(a) for a in DATA.IDEA_TPL["acts"]]
        out += ["", "**反转**：" + DATA.IDEA_TPL["twist"]]
        out += ["", "**开篇试写**", DATA.IDEA_TPL["hook"].format(name)]
        return "\n".join(out)

    def _idea_subject(self, text):
        t = text or ""
        for w in ("给我", "帮我", "来一个", "来一", "想一个", "想个", "写一个", "写个",
                  "一个", "一份", "一本", "一点", "小说", "故事", "剧情", "剧本",
                  "的", "灵感", "构思", "创意", "点子", "构想", "大纲", "题材", "写什么"):
            t = t.replace(w, " ")
        t = re.sub(r"[，。！？、,.；;：:!?\s]+", " ", t).strip()
        t = t.split(" ")[0].strip() if t else ""
        return t if 1 <= len(t) <= 10 else ""

    def _studio_route(self, text):
        scoped = self._studio_scope(text)
        name_hits = self._studio_novel_hits(text)
        novel_kw = ["小说", "作品", "书", "系列", "番外", "阅读", "追更", "连载",
                    "在哪看", "哪里看", "看网址", "我该看", "趣生活"]
        # 排除「让我写小说」这类创作请求, 那是作文/代码的活, 不是查工作室书目
        make_verb = ["写一篇", "写个", "写本", "帮我写", "创作", "生成一", "来一篇", "编一个", "改写",
                     "灵感", "创意", "构思", "点子", "诗", "诗歌", "小游戏", "设计", "大纲", "写什么"]
        ask_list = ["有哪些", "几本", "几部", "都有什么", "有什么", "推荐", "网址", "在哪看",
                    "哪里看", "系列", "介绍下", "介绍一下"]
        want_novel = bool(name_hits) or ((scoped or any(k in text for k in novel_kw))
                                         and any(k in text for k in novel_kw))
        if want_novel and any(v in text for v in make_verb) and not any(a in text for a in ask_list):
            return None
        if want_novel:
            return self._studio_novels_answer(text, name_hits)
        if not scoped:
            return None
        # 单点事实: 只答命中分最高的那一条, 绝不把相邻事实一起倒出来
        best, best_score = None, 0
        for f in DATA.STUDIO_FACTS:
            s = sum(1 for kw in f["kws"] if kw in text)
            if s > best_score:
                best, best_score = f, s
        if best is not None:
            return best["body"]
        # 泛问「小方工作室是什么 / 介绍一下」-> 一段名片式简介
        return ("**{}**（{}）是一个 {}。\n\n"
                "- 口号：{}\n- 理念：{}\n- 核心成员：{}\n- 纪念日：{}\n"
                "- 官网：{}（AI 小方专站 {}）\n- 技术栈：{}\n- 官方账号：{}").format(
                    DATA.STUDIO_NAME, DATA.STUDIO_EN, DATA.STUDIO_KIND,
                    DATA.STUDIO_SLOGAN, DATA.STUDIO_IDEA, DATA.STUDIO_MEMBERS,
                    DATA.STUDIO_ANNIVERSARY, DATA.STUDIO_SITE, DATA.STUDIO_SITE_AI,
                    DATA.STUDIO_TECH, DATA.STUDIO_SOCIAL)

    # ===============================================================
    # v1.7 Alpha: 其他地区时间 / 当前位置 —— 本机时区表白算, 不惊动联网
    # ===============================================================
    _CITY_TZ = [
        ("纽约", -5.0, ["纽约", "new york"]), ("洛杉矶", -8.0, ["洛杉矶", "旧金山", "硅谷", "西雅图", "los angeles"]),
        ("伦敦", 0.0, ["伦敦", "london"]), ("巴黎", 1.0, ["巴黎", "paris"]),
        ("柏林", 1.0, ["柏林", "berlin"]), ("莫斯科", 3.0, ["莫斯科", "moscow"]),
        ("迪拜", 4.0, ["迪拜", "dubai"]), ("新德里", 5.5, ["新德里", "孟买", "印度"]),
        ("曼谷", 7.0, ["曼谷", "bangkok"]), ("新加坡", 8.0, ["新加坡", "singapore"]),
        ("北京", 8.0, ["北京", "上海", "广州", "深圳", "香港", "台北"]),
        ("东京", 9.0, ["东京", "大阪", "日本", "tokyo"]), ("首尔", 9.0, ["首尔", "韩国", "seoul"]),
        ("悉尼", 10.0, ["悉尼", "墨尔本", "澳大利亚", "sydney"]),
    ]

    def _city_time(self, text):
        """v1.7: 问"某地现在几点" —— 用本地时区表换算(标准时间, 未计夏令时), 不给假数据。"""
        t = (text or "").lower()
        if not any(k in t for k in ["几点", "时间", "现在"]):
            return None
        for city, off, kws in self._CITY_TZ:
            if any(k in t for k in kws):
                now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=off)
                sign = "+" if off >= 0 else "-"
                return "🕐 {} 现在是 {}（UTC{}{:g}，标准时间，未计夏令时）。".format(
                    city, now.strftime('%H:%M:%S'), sign, abs(off))
        return None

    def _location_answer(self, text):
        """v1.7: 问"我在哪/当前位置" —— 用本机时区如实告知大概区域(不做假定位), 并给出更准的下一步。"""
        t = (text or "")
        if not any(k in t for k in ["我在哪", "我在哪里", "我的位置", "当前位置", "我的定位",
                                    "我在什么城市", "我在哪个城市", "我在的城市", "我在哪儿"]):
            return None
        try:
            off = -time.timezone / 3600.0
            tz = time.tzname[0] if time.tzname else ""
        except Exception:
            off, tz = 8.0, ""
        sign = "+" if off >= 0 else "-"
        return ("📍 我不能精确定位你的坐标（那需要联网或 GPS 授权），但从这台设备的时区看，"
                "你现在位于 **UTC{}{:g}**{} 一带。想看更准的，直接报个城市名，我马上帮你查。").format(
            sign, abs(off), ("（{}）".format(tz) if tz else ""))

    # ===============================================================
    # v1.7 Alpha: 实时天气 —— 直连真实气象数据源(wttr.in, 免费无密钥)
    #   覆盖"当前天气 / 未来 3 天 / 未来 24 小时"三档; 拉不到就如实回落联网检索,
    #   绝不编造温度 —— 天气是最不能瞎猜的一类问题。
    # ===============================================================
    _WX_KWS = ["天气", "气温", "温度", "预报", "下雨", "下雪"]
    # 出现在"写作/创作"语境里的"天气"只是素材, 不许抢答
    _WX_NOT_ASK = ["写一篇", "写个", "写一段", "写首", "作文", "散文", "小说",
                   "诗", "代码", "程序", "剧本", "灵感", "构思", "帮我写"]
    # 天气专用城市表 —— 不能复用 _CITY_TZ: 那边的关键词是"时区同区分组"
    #   (上海/广州/深圳 都并进北京), 拿来查天气会把"上海天气"答成北京天气。
    _WX_CITIES = [
        ("北京", ["北京"]), ("上海", ["上海"]), ("广州", ["广州"]), ("深圳", ["深圳"]),
        ("香港", ["香港"]), ("台北", ["台北"]), ("澳门", ["澳门"]),
        ("杭州", ["杭州"]), ("南京", ["南京"]), ("苏州", ["苏州"]), ("成都", ["成都"]),
        ("重庆", ["重庆"]), ("武汉", ["武汉"]), ("西安", ["西安"]), ("天津", ["天津"]),
        ("青岛", ["青岛"]), ("厦门", ["厦门"]), ("长沙", ["长沙"]), ("郑州", ["郑州"]),
        ("济南", ["济南"]), ("沈阳", ["沈阳"]), ("大连", ["大连"]), ("哈尔滨", ["哈尔滨"]),
        ("昆明", ["昆明"]), ("福州", ["福州"]), ("合肥", ["合肥"]), ("南昌", ["南昌"]),
        ("贵阳", ["贵阳"]), ("南宁", ["南宁"]), ("兰州", ["兰州"]), ("乌鲁木齐", ["乌鲁木齐"]),
        ("拉萨", ["拉萨"]), ("呼和浩特", ["呼和浩特"]), ("银川", ["银川"]), ("西宁", ["西宁"]),
        ("海口", ["海口"]), ("三亚", ["三亚"]), ("无锡", ["无锡"]), ("宁波", ["宁波"]),
        ("佛山", ["佛山"]), ("东莞", ["东莞"]), ("珠海", ["珠海"]), ("温州", ["温州"]),
        ("东京", ["东京", "tokyo"]), ("大阪", ["大阪", "osaka"]), ("京都", ["京都", "kyoto"]),
        ("首尔", ["首尔", "seoul"]), ("新加坡", ["新加坡", "singapore"]),
        ("曼谷", ["曼谷", "bangkok"]), ("吉隆坡", ["吉隆坡"]), ("雅加达", ["雅加达"]),
        ("悉尼", ["悉尼", "sydney"]), ("墨尔本", ["墨尔本", "melbourne"]), ("奥克兰", ["奥克兰"]),
        ("纽约", ["纽约", "new york"]), ("洛杉矶", ["洛杉矶", "los angeles"]),
        ("旧金山", ["旧金山"]), ("西雅图", ["西雅图", "seattle"]), ("芝加哥", ["芝加哥"]),
        ("波士顿", ["波士顿"]), ("华盛顿", ["华盛顿"]), ("多伦多", ["多伦多"]),
        ("温哥华", ["温哥华"]), ("伦敦", ["伦敦", "london"]), ("巴黎", ["巴黎", "paris"]),
        ("柏林", ["柏林", "berlin"]), ("阿姆斯特丹", ["阿姆斯特丹"]), ("苏黎世", ["苏黎世"]),
        ("罗马", ["罗马"]), ("马德里", ["马德里"]), ("莫斯科", ["莫斯科", "moscow"]),
        ("迪拜", ["迪拜", "dubai"]), ("新德里", ["新德里", "德里"]), ("孟买", ["孟买"]),
    ]
    # 兜底正则里要排除的"假城市"(时间/量词/方位词)
    _WX_STOP = {"小时", "分钟", "今天", "明天", "后天", "昨天", "未来", "现在", "最近",
                "这几", "这几天", "这里", "那边", "当地", "本地", "外面", "天气", "气温",
                "温度", "预报", "三天", "两天", "一周", "几天", "半天", "最高", "最低",
                "上午", "下午", "晚上", "早上", "中午", "夜间", "白天", "全天", "下周",
                "这个", "那个", "哪个", "什么", "怎么", "如何", "多少"}

    def _weather_city(self, text):
        """从问句里抠出城市名: 先查天气专用城市表(长名优先), 再按"XX天气"句式兜底。"""
        t = (text or "").lower()
        # 长名优先, 避免 "德里" 抢在 "新德里" 前面
        for city, kws in sorted(self._WX_CITIES, key=lambda kv: -max(len(k) for k in kv[1])):
            if any(k in t for k in kws):
                return city
        m = re.search(r"([\u4e00-\u9fa5]{2,5})(?:市|的)?(?:今天|明天|后天|未来|最近|这几天)?"
                      r"(?:天气|气温|温度|预报)", text or "")
        if m:
            cand = m.group(1)
            if cand not in self._WX_STOP:
                return cand
        return ""

    def _weather_fetch(self, city):
        """拉实时气象 JSON; 网络卡死由 _run_with_timeout 硬超时兜底, 不阻塞主流程。"""
        def _go():
            import requests
            r = requests.get("https://wttr.in/" + (city or ""),
                             params={"format": "j1", "lang": "zh"},
                             timeout=10,
                             headers={"User-Agent": "curl/8"})
            if r.status_code != 200:
                return None
            return r.json()
        return _run_with_timeout(_go, timeout=12.0, default=None)

    @staticmethod
    def _wx_area(data, city):
        na = data.get("nearest_area") or []
        if na:
            area = (na[0].get("areaName") or [{}])[0].get("value", "")
            region = (na[0].get("region") or [{}])[0].get("value", "")
            if region and region != area:
                area = "{} {}".format(region, area)
            if area:
                return area
        return city or "当前位置"

    def _weather_answer(self, text):
        t = text or ""
        if not any(k in t for k in self._WX_KWS):
            return None
        if any(k in t for k in self._WX_NOT_ASK):
            return None
        city = self._weather_city(t)
        want_hour = any(k in t for k in ["24小时", "24 小时", "每小时", "逐小时", "分时"])
        want_days = (not want_hour) and any(k in t for k in ["三天", "3天", "未来", "后几天", "这几天"])
        data = self._weather_fetch(city)
        if not isinstance(data, dict):
            return None
        try:
            cur = (data.get("current_condition") or [{}])[0]
            days = data.get("weather") or []
            area = self._wx_area(data, city)
            obs = (cur.get("observation_time") or "").strip()
            obs = obs.replace("AM", "").replace("PM", "").strip() or \
                datetime.datetime.now().strftime("%H:%M")

            if want_hour:
                return self._wx_hourly(area, days, cur, obs)
            if want_days:
                return self._wx_daily(area, days, obs)

            desc = _wcode_zh(cur.get("weatherCode"),
                             (cur.get("weatherDesc") or [{}])[0].get("value", ""))
            wind = "{} {} km/h".format(cur.get("winddir16Point", ""),
                                       cur.get("windspeedKmph", "")).strip()
            today = days[0] if days else {}
            tmax, tmin = today.get("maxtempC", "-"), today.get("mintempC", "-")
            rain = ""
            try:
                hr = today.get("hourly") or []
                if hr:
                    rain = max(int(h.get("chanceofrain", 0)) for h in hr)
            except Exception:
                rain = ""
            out = ["🌤️ **{} 当前天气**".format(area), "",
                   "| 项目 | 数值 |", "| :-- | :-- |",
                   "| 天气 | **{}** |".format(desc),
                   "| 气温 | **{} ℃**（体感 {} ℃） |".format(cur.get("temp_C", "-"), cur.get("FeelsLikeC", "-")),
                   "| 今日区间 | {} ℃ ~ {} ℃ |".format(tmin, tmax),
                   "| 湿度 | {} % |".format(cur.get("humidity", "-")),
                   "| 降水概率 | {} % |".format(rain if rain != "" else "-"),
                   "| 风 | {} |".format(wind or "-"),
                   "| 能见度 | {} km |".format(cur.get("visibility", "-")),
                   "| 紫外线 | {} |".format(cur.get("uvIndex", "-")),
                   "| 观测时间 | {} |".format(obs),
                   "", "> 数据源：wttr.in 实时气象接口（{}）。".format(area)]
            return "\n".join(out)
        except Exception:
            return None

    def _wx_daily(self, area, days, obs):
        if not days:
            return None
        out = ["📅 **{} 未来 3 天天气**".format(area), "",
               "| 日期 | 天气 | 最高 | 最低 | 降水概率 | 风速 |", "| :-- | :-- | --: | --: | --: | --: |"]
        for d in days[:3]:
            hr = d.get("hourly") or []
            mid = hr[4] if len(hr) > 4 else (hr[0] if hr else {})
            desc = _wcode_zh(mid.get("weatherCode"),
                             (mid.get("weatherDesc") or [{}])[0].get("value", ""))
            try:
                rain = max(int(h.get("chanceofrain", 0)) for h in hr) if hr else 0
            except Exception:
                rain = 0
            try:
                wmax = max(int(h.get("windspeedKmph", 0)) for h in hr) if hr else 0
            except Exception:
                wmax = 0
            wk = "一二三四五六日"
            try:
                _d = datetime.datetime.strptime(d.get("date", ""), "%Y-%m-%d")
                label = "{} 周{}".format(_d.strftime("%m-%d"), wk[_d.weekday()])
            except Exception:
                label = d.get("date", "-")
            out.append("| {} | {} | {} ℃ | {} ℃ | {} % | {} km/h |".format(
                label, desc, d.get("maxtempC", "-"), d.get("mintempC", "-"), rain, wmax))
        out += ["", "> 数据源：wttr.in 实时气象接口（{}），观测时间 {}。".format(area, obs)]
        return "\n".join(out)

    def _wx_hourly(self, area, days, cur, obs):
        rows = []
        try:
            now_h = datetime.datetime.now().hour
            for di, d in enumerate(days[:2]):
                for h in (d.get("hourly") or []):
                    hh = int(h.get("time", 0)) // 100
                    if di == 0 and hh < now_h:
                        continue
                    rows.append((d.get("date", ""), hh, h))
        except Exception:
            rows = []
        if not rows:
            return None
        rows = rows[:8]
        out = ["⏱️ **{} 未来 24 小时天气**".format(area), "",
               "| 时间 | 天气 | 气温 | 体感 | 降水概率 | 风速 |", "| :-- | :-- | --: | --: | --: | --: |"]
        for date, hh, h in rows:
            desc = _wcode_zh(h.get("weatherCode"),
                             (h.get("weatherDesc") or [{}])[0].get("value", ""))
            out.append("| {} {:02d}:00 | {} | {} ℃ | {} ℃ | {} % | {} km/h |".format(
                date[5:], hh, desc, h.get("tempC", "-"), h.get("FeelsLikeC", "-"),
                h.get("chanceofrain", "-"), h.get("windspeedKmph", "-")))
        out += ["", "> 数据源：wttr.in 实时气象接口（{}），观测时间 {}。".format(area, obs)]
        return "\n".join(out)

    # v1.7 Alpha: 情感优先通道 —— 先把情绪接住, 再谈别的
    #   旧版问题: "我好开心啊今天" 被"今天"的日期分支抢答; "我最近好难过" 掉进知识库倒条目。
    #   现在: 情绪词 + 第一人称 → 直接给共情/共鸣, 不翻库、不报日期、不倒模板。
    _EMO_WORDS = ["开心", "高兴", "快乐", "幸福", "兴奋", "激动", "满足", "感动", "惊喜",
                  "难过", "伤心", "委屈", "焦虑", "紧张", "害怕", "恐惧", "压力", "累",
                  "烦躁", "烦躁", "生气", "愤怒", "气死", "崩溃", "孤独", "寂寞", "失落",
                  "郁闷", "沮丧", "绝望", "无助", "失眠", "痛苦", "无聊", "烦", "emo", "EMO"]
    _EMO_SELF = ["我", "咱", "自己", "心情", "心里", "感受", "情绪", "最近", "今天", "真的", "感觉"]
    _EMO_TASK = ["写代码", "写程序", "写个", "写一个", "作文", "散文", "小说", "诗", "代码",
                 "编程", "实现", "计算", "算一下", "解方程", "方程", "微分", "积分", "导数",
                 "天气", "气温", "几点", "几号", "星期", "日期", "推荐", "翻译", "查一下", "搜索"]

    def _emotion_reply(self, text, emo):
        t = text or ""
        if len(t) < 3 or any(k in t for k in self._EMO_TASK):
            return None
        score = emo.get("score", 0)
        cat = emo.get("category", "")
        # 触发条件: 有情绪词, 或情绪分类明显偏离中性(此时 score 也要够看)
        emotiony = any(k in t for k in self._EMO_WORDS) or \
            (cat not in ("", "完全中性") and abs(score) >= 1.0)
        if not emotiony:
            return None
        if not any(k in t for k in self._EMO_SELF):
            return None
        if score <= -1.0 or any(emo.get(k) for k in ("has_sad", "has_anger", "has_fear", "has_tired")):
            body = self.generator._gen_comfort(emo, None)
            return body + " 想具体说说发生什么了吗？我听着，不着急。"
        if score >= 1.5:
            return "听到你开心，我也跟着高兴 😊 这份好心情值得好好记一笔。" \
                   "发生什么好事了？说来让我也乐一乐。"
        return None

    def reply(self, user_input, emo, intent, kb_hits):
        text = user_input.strip()
        # v1.7 Alpha: 创作类综合任务(写诗 / 设计小游戏 / 小说灵感) —— 现场成篇, 先于一切检索
        _cv = self._creative_route(text)
        if _cv:
            return _cv
        # v1.7 Alpha: 小方工作室精准靶向 —— 先于一切通用逻辑, 问一件只答一件, 全程本地
        _studio = self._studio_route(text)
        if _studio:
            return _studio
        # v0.5 Alpha: 官网/网站类问题只答官网 (不答模型身份)
        site_ask = ("官网" in text) or ("fanggame" in text) or ("fanggame.company" in text) or \
            ("网址" in text and "工作室" in text)
        if not site_ask and "网站" in text and any(m in text for m in
                ["小方", "工作室", "fanggame", "网站是多少", "官网是多少"]):
            site_ask = True
        # v0.6 Pro: 要网址就只给网址 —— 不顺手介绍自己是啥/工作室是谁 (防止曲解意图)
        if site_ask:
            return "小方工作室官网：{}".format(STUDIO_SITE)
        # v1.7 Alpha: 其他地区时间 / 当前位置 —— 本地算得出, 直接答, 不必联网
        _ct = self._city_time(text)
        if _ct:
            return _ct
        _loc = self._location_answer(text)
        if _loc:
            return _loc
        # v1.7 Alpha: 实时天气 —— 直连气象数据源, 问一件答一件(当前/3天/24小时), 不编温度
        _wx = self._weather_answer(text)
        if _wx:
            return _wx
        # v1.7 Alpha: 多轮投稿 —— 「再改改 / 换算法 / 加长 / 改结尾」接着上一版继续, 不重出原文
        if self._is_revise(text) and (self.last_essay or self.last_code):
            _rev = self._revise_last(text, emo, kb_hits)
            if _rev:
                return _rev
        # v1.7 Alpha: 情感优先 —— 情绪明显时先接住情绪(共情/共鸣), 不翻库、不倒模板、不报日期
        _emo_r = self._emotion_reply(text, emo)
        if _emo_r:
            return _emo_r
        # v1.5 正式版: 询问机制 —— 意图不明/缺关键参数(如代码没给语言、作文没给主题)先反问
        #   选项数量与内容由 AI 自定, 末项恒为「其他」, 支持单选/多选; 问完等用户回答。
        _a = self._maybe_ask(text, intent)
        if _a:
            return _a
        # v1.1 Alpha: 长线作文/成文最优先 —— 写一篇/作文/围绕X写 → 稳定长文, 不被时/日/身份等抢答
        if self._is_essay(text):
            _ess = self._gen_essay(text, emo, kb_hits)
            _tp, _ = self._extract_topic(text)
            # v1.7: 记住这一版, 供"多轮投稿"接着改
            self.last_essay = {"topic": "" if _tp in ("这个话题", "这件事") else _tp,
                               "text": _ess, "round": 1}
            self.last_kind = "essay"
            return _ess
        # v0.5 Alpha 2: 设定人格——仅在明确问生活/身世时启用, 否则保持中性 AI 助手
        if intent["top"] == "persona":
            pr = self.persona.reply(text)
            if pr:
                return pr
        # v1.5 Alpha: 系统提示词/工作守则类提问 → 交底(身份/厂商/聊天/情感/写码/计算/终止/达标)
        if any(m in text for m in ["系统提示词", "你的规则", "你的规范", "你的纪律", "你的提示词",
                                   "你怎么工作", "你的原则", "你的守则", "system prompt",
                                   "你是什么规则", "你按什么标准"]):
            return _system_prompt_answer()
        # v0.5 正式版: 数学 / 代码
        if intent["top"] == "math":
            ma = self.math.answer(text)
            if ma:
                return _terminate_clean(ma)
        if intent["top"] == "code":
            ca = self.code.reply(text)
            if ca:
                # v1.7: 记住这一版代码, 供"多轮投稿"接着改
                self.last_code = {"req": text, "lang": self.code._detect_lang(text) or "Python",
                                  "text": ca, "round": 1}
                self.last_kind = "code"
                return _terminate_clean(ca)
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
        # v1.7 Alpha: 情绪明显时不能被"今天"的日期分支抢答 ——
        #   "我好开心啊今天" 里的"今天"只是语气词, 用户要的是共情而不是报日期。
        #   只有当用户真的在问日期(几号/星期几/日期/什么日子)时才走日期分支。
        _emo_strong = abs(emo.get("score", 0)) >= 2 and not any(
            w in text for w in ["几号", "几月", "星期几", "周几", "日期", "哪一天", "哪天",
                                "什么日子", "日历", "多少号", "几点了"])
        if intent["top"] == "date" and intent["max_score"] >= 1 and not _emo_strong:
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
        ans = self.generator.generate(text, kb_hits, emo, intent, self.history)
        # v1.5 Alpha: 达标检测 + 终止判断 —— 答非所问/空答回退; 缺终止符补句读
        if not _meets_std(ans):
            return "嗯…这条我一时没想好怎么答最有把握，你再具体一点，我马上认真回你。"
        return _terminate_clean(ans)

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
        """v0.6 多轮思考: 小方边答边想/边想边答 (类 DeepSeek V4 Pro)。
        v1.7 Alpha 重写: 不再把句子按 i%N 交错重排(那会把长文彻底打乱、Markdown 全废),
        改为【顺序切段】; 含 Markdown 结构(标题/列表/表格/代码块)时整篇按原样顺序输出,
        绝不重排、绝不截断。"""
        # 含 Markdown 结构 → 顺序完整输出, 不做任何重排
        if re.search(r"(^|\n)\s*(#{1,6}\s|```|\|.*\||\s*[-*+]\s|\s*\d+\.\s)", answer):
            _typewrite("小方: " + answer)
            return
        _typewrite_lines(["🧠 多轮思考进行中… 我分 {} 步把答案想完整。".format(THINK_TURNS)], C_DEEP)
        sents = [s for s in re.split(r"(?<=[。！？.!?])", answer) if s.strip()]
        if not sents:
            _typewrite("小方: " + answer)
            return
        total = max(1, THINK_TURNS)
        # —— 顺序切段(连续分块), 保持原文语序 ——
        n = len(sents)
        size = max(1, -(-n // total))
        groups = [sents[i:i + size] for i in range(0, n, size)]
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
            print(C_REPLY + "🧠 v1.7 Alpha 超大型更新: 2B+ 大模型(≈{:.1f}亿参数·{}层纵深·int8量化常驻≈{:.2f}GB·自动降级算力不减)".format(
                MODEL_PARAMS / 1e8, MODEL_LAYERS, _TIER_GB) + C_RESET)
            print(C_REPLY + "🎓 学习: 初始权重不再固定 —— fp32 主权重 + AdamW 优化器 + 反向传播, 每轮对话真在做梯度更新" + C_RESET)
            print(C_REPLY + "🧩 深度思考: 两层嵌套(思考→验算→再思考→再验算) 由 Transformer 自己生成, 收在终止符 {} 才落笔".format(TERMINATOR) + C_RESET)
            print(C_REPLY + "📐 尺度: {} (想上 3B/4B: 设 XIAOFANG_SCALE=3b/4b, 内存不足会自动安全回落)".format(SCALE_NOTE) + C_RESET)
            print(C_REPLY + "🔎 检索: 数据库优先 → 库里没有再联网 → 进网页抠正文整合 | 代码 18 门语言真实现 | 作文按题目/字数/要点定向成篇" + C_RESET)
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
        if UI_ST.get("live"):         # v1.4: 思考实时打字结束后, 打一条最终状态行(状态面板定点)
            sys.stdout.write(C_HINT + "  ⟳ " + _ui_status_text() + C_RESET + "\n")
            sys.stdout.flush()
            UI_ST["last_tick"] = time.time()
        # —— 思考完才 "再检测关键词" ——
        light_chat = intent["top"] in ("greet", "bye", "thanks", "joke", "help",
                                       "time", "date", "smalltalk") or len(text) <= 6
        preset_answer = None
        # v1.7 Alpha: 小方工作室精准靶向 —— 命中就绕过预设关键词, 保证"问一件只答一件"
        _studio_pre = self._studio_route(text)
        if light_chat and not _studio_pre:
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
            self._train_turn(text, answer)   # v1.7: 闲聊也做一步反向传播
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
        if THINK_MODE == "multi" and SHOW_DEEP_THINK and not _studio_pre:
            self._multi_deliver(answer, emo, intent, kb_hits)
        else:
            _typewrite("小方: " + answer)
        print(C_HINT + "  [{}]".format(self.meter.format()) + C_RESET)
        self.history.append(text)
        self.history.append(answer)
        self.lm.add_text(text)
        self.lm.add_text(answer)
        self._train_turn(text, answer)   # v1.7: 每轮真实反向传播 + AdamW 更新权重
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

    # ══════════════════════════════════════════════════════════════════════
    # v1.7 Alpha · 在线学习: 每轮对话都做一次真正的反向传播 + AdamW 更新
    #   → Transformer 的初始权重不再是"一次性固定"的, 而是随对话持续被优化器改写。
    #   监督目标是 next-token: 让模型学着"给定用户这句, 该接出怎样的回答"。
    # ══════════════════════════════════════════════════════════════════════
    def _train_turn(self, text, answer=None):
        """v1.7 Alpha: 一轮在线反向传播。返回 {"loss","gnorm","steps"} 或 None。"""
        if not TRAIN_ENABLED:
            return None
        tr = getattr(self, "transformer", None)
        if tr is None or not getattr(tr, "bank", None):
            return None
        try:
            toks = self.tokenizer.tokenize(text or "")
            if answer:
                toks = toks + [TERMINATOR] + self.tokenizer.tokenize(answer)
            toks = toks[:TRAIN_MAX_SEQ]
            if len(toks) < 4:
                return None
            ids = [tr.token2id.get(t, 0) for t in toks]
            if len(set(ids)) < 2:
                return None
            info = None
            for _ in range(max(1, int(TRAIN_STEPS_PER_TURN))):
                info = tr.train_step(ids[:-1], ids[1:]) or info
            # 每 N 轮落盘一次, 免得每轮都写盘拖慢速度
            self._train_turns = getattr(self, "_train_turns", 0) + 1
            if self._train_turns % max(1, int(TRAIN_SAVE_EVERY)) == 0:
                tr.dump_train_state()
            self._last_train = info
            return info
        except Exception:
            return None

    def _train_brief(self):
        """v1.7 Alpha: 给深度思考面板用的一行训练/优化器摘要。"""
        tr = getattr(self, "transformer", None)
        if tr is None or not getattr(tr, "bank", None):
            return None
        try:
            st = tr.train_stats()
            hist = st.get("hist") or []
            h = " → ".join("{:.4f}".format(float(v)) for v in hist[-4:]) if hist else "—"
            return ("在线学习(反向传播): 累计{} 步 · 本步loss={} · ‖∇‖={} · 参数量={:,}个fp32主权重".
                    format(st.get("total", 0),
                           "{:.4f}".format(st.get("loss", 0.0)) if st.get("loss") else "—",
                           "{:.4f}".format(st.get("gnorm", 0.0)) if st.get("gnorm") else "—",
                           sum(int(getattr(a, "size", 0)) for _n, a in tr.bank.params)))
        except Exception:
            return None

    def _train_hist_line(self):
        """v1.7 Alpha: 近若干步 loss 曲线(供冒烟测试/思考面板打印)。"""
        tr = getattr(self, "transformer", None)
        if tr is None or not getattr(tr, "bank", None):
            return "—"
        try:
            hist = tr.train_stats().get("hist") or []
            return " ".join("{:.4f}".format(float(v)) for v in hist) if hist else "—"
        except Exception:
            return "—"

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
            # —— v1.4: 思考期 → 折叠/缓冲模式下才原地重绘面板; 默认(live)思考直接打字, 不抢屏 ——
            if UI_ST["capture"] and not UI_ST.get("live"):
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
            # v1.7 Alpha: 改成"无输出静默"判卡 —— 只要还有任何新输出(打字机/面板/阶段),
            #   就说明它活着, 长篇回答不会在打到一半被砍断。
            _last_out = max(t0, float(_OUT_TICK.get("t") or 0.0), float(UI_ST.get("last_tick") or 0.0))
            if (now - _last_out > RESPONSE_TIMEOUT) or (now - t0 > ABSOLUTE_TURN_CAP):
                sys.stdout.write("\r" + " " * 70 + "\r")
                sys.stdout.flush()
                self._print_note(
                    "\n⚠ 已连续 {} 秒没有任何新输出，可能是真卡住了(联网/网页读取最容易卡)。小方先让开，"
                    "你可以直接输入新消息，或输入 /stop 打断它。".format(RESPONSE_TIMEOUT), C_DEEP)
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