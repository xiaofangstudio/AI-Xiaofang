# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════════
# svgdream_covi1.py  V2.8
#   通用 SVG + CSS 图像生成引擎(从小方主程序独立而来, 自成一体, 纯标准库/零第三方)。
#   · 不再只会画鹈鹕 —— 一个能理解「你想画什么 + 什么气氛」的动态矢量画引擎。
#   · 每请求流程: 读意图(主体/时间/天气/情绪/颜色/挂件) → 中文画面 + 英文真图 Prompt
#             → 几何部件 + CSS 动画现场生成会动的 SVG → 落盘 .svg + .prompt.txt。
#   · 对外统一入口: svgdream_answer(text) / 兼容旧名 pelican_answer(text)。
#   独立 import 本文件不加载任何神经网络权重、不占显存 —— 可单独跑/单独测。
# ══════════════════════════════════════════════════════════════════════════════

# v2.4 收口: 鹈鹕引擎彻底内联进主程序 —— 根目录不再留任何独立引擎文件。
#   之前它虽然走 pelican_answer() 统一入口, 但实现仍是根目录下单独一个
#   pelican_bike_covi1.py, 外人打开项目会纳闷"怎么还专门有个测鹈鹕的"。
#   现在整段实现(配色 / 几何 / SVG 模板 / 语义解析 / 出图 / 落盘)全部内联在下面,
#   零第三方依赖(纯标准库)。命中「鹈鹕骑自行车 / 鹈鹕猛猛蹬」就现场生成一张会动的
#   SVG 并落盘, 参数每次重新随机(张张不重样), 传整数 seed 可复现同一条。
#   对外唯一入口依旧是 pelican_answer(), 所有调用点零改动。
# ============================================================

# 鹈鹕引擎（原 pelican_bike_covi1.py 整段内联在此，已无同名文件）——
# 「鹈鹕猛猛蹬」海边骑自行车 动态 SVG 生成器。C1 全系（Lite / Pro / Ultra）共用。
# 零第三方依赖，只需要标准库。
#
# 对外只暴露五件事:
#     is_pelican_bike(text)              -> 判断这句话是不是要"鹈鹕骑自行车"
#     parse_constraints(text)            -> 解析附加要求（篮子 / 颜色 / 戴什么）
#     pelican_bike_svg(seed, info)       -> 生成一段完整 <svg>（真 CSS 动画）
#     pelican_bike_save(svg, info)       -> 落盘成 .svg 文件, 返回绝对路径
#     pelican_bike_reply(text, seed)     -> 一步到位, 直接给出可以直接回给用户的文本
#
# 每次调用都会重新随机（配色 / 辐条 / 挂件 / 车筐里的鱼 / 踏频 / 浪速 / 云 / 海鸥 / 尘土 / 拟声词 …）,
# 所以绝不会两次输出一模一样。传整数 seed 可复现同一条。

# 这一段内联代码只用标准库这五个(文件顶部本来就已导入, 这里再写一遍是为了
# 让它自成一体 —— 单独抽出来也能跑, 不依赖主程序顶部的导入)。
import os
import re
import math
import random
import datetime


# ──────────────────────────────────────────────────────────────────────────────
# 配色（天空 3 段 / 海 2 段 / 沙 2 段 / 太阳 / 太阳芯 / 影子）
# ──────────────────────────────────────────────────────────────────────────────
_PB_PALETTES = [
    {"name": "清晨", "sky": ("#bfe6ff", "#eaf6ff", "#fff6e0"), "sea": ("#3f8fc4", "#1f5f92"),
     "sand": ("#f4e0b8", "#e3c894"), "sun": "#ffd98a", "core": "#fff3c4", "shadow": "#b99a63"},
    {"name": "正午", "sky": ("#8fd3ff", "#cbeaff", "#f2fbff"), "sea": ("#2f86c8", "#14517f"),
     "sand": ("#f7e6bd", "#e6cd97"), "sun": "#ffe08a", "core": "#fff8d2", "shadow": "#b99a63"},
    {"name": "黄昏", "sky": ("#ffb27a", "#ffd6a5", "#ffe9c9"), "sea": ("#d2704f", "#7a3f46"),
     "sand": ("#f0c9a0", "#d9a878"), "sun": "#ff8a4c", "core": "#ffe2a6", "shadow": "#a06a45"},
    {"name": "傍晚", "sky": ("#6f7fd6", "#9fb0ea", "#ffd9c2"), "sea": ("#2b4a86", "#16294f"),
     "sand": ("#cbb5a2", "#a89484"), "sun": "#ffb27a", "core": "#ffe6c2", "shadow": "#6b5a4c"},
    {"name": "夜色", "sky": ("#131a3a", "#243063", "#4a5aa0"), "sea": ("#1a2a55", "#0b1430"),
     "sand": ("#5d5f7a", "#3e4058"), "sun": "#f2f5ff", "core": "#ffffff", "shadow": "#1b1e30"},
]

_PB_ACC_DESC = {
    "": "什么也没戴",
    "cap": "头顶扣了顶小凉帽",
    "scarf": "脖子上绕了条围巾",
    "shades": "架着副墨镜",
    "flower": "羽毛上别了朵小花",
}

_PB_SFX = ["嗖——", "哗啦", "猛猛蹬", "蹬！蹬！", "呼——", "海在往后退"]

_FONT = "Noto Sans CJK SC, Microsoft YaHei, PingFang SC, sans-serif"

# 几何常量（画布 720 x 440，地面 y=340）
_WY = 306          # 轮轴高度
_RWX, _FWX = 302, 446
_WR = 34           # 轮半径
_BBX, _BBY, _CR = 374, 310, 24   # 五通 + 曲柄半径
_HIPX, _HIPY = 352, 250          # 鹈鹕屁股（坐垫）

# ──────────────────────────────────────────────────────────────────────────────
# 模板
# ──────────────────────────────────────────────────────────────────────────────
_PB_SVG_TPL = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440" width="720" height="440" role="img" aria-label="鹈鹕在海边骑自行车">
  <defs>
    <linearGradient id="pbSky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="@SKY1@"/><stop offset="0.64" stop-color="@SKY2@"/><stop offset="1" stop-color="@SKY3@"/>
    </linearGradient>
    <linearGradient id="pbSea" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="@SEA1@"/><stop offset="1" stop-color="@SEA2@"/>
    </linearGradient>
    <linearGradient id="pbSand" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="@SAND1@"/><stop offset="1" stop-color="@SAND2@"/>
    </linearGradient>
    <radialGradient id="pbSunG">
      <stop offset="0" stop-color="@SUNCORE@" stop-opacity="0.95"/>
      <stop offset="0.55" stop-color="@SUN@" stop-opacity="0.45"/>
      <stop offset="1" stop-color="@SUN@" stop-opacity="0"/>
    </radialGradient>
    <style>
      .pbSpinA{animation:pbSpinK @WHEELT@s linear infinite;transform-box:fill-box;transform-origin:50% 50%;animation-delay:@DLW@}
      .pbSpinB{animation:pbSpinK @WHEELT@s linear infinite;transform-box:fill-box;transform-origin:50% 50%;animation-delay:@DLW@}
      .pbCrank{animation:pbCrankK @CRANKT@s linear infinite;transform-box:fill-box;transform-origin:50% 50%;animation-delay:@DLC@}
      .pbBob{animation:pbBobK @BOBT@s ease-in-out infinite;animation-delay:@DLB@}
      .pbLegA{animation:pbLegAK @CRANKT@s ease-in-out infinite;transform-box:fill-box;transform-origin:50% 50%;animation-delay:@DLC@}
      .pbLegB{animation:pbLegBK @CRANKT@s ease-in-out infinite;transform-box:fill-box;transform-origin:50% 50%;animation-delay:@DLC@}
      .pbWing{animation:pbWingK @WINGT@s ease-in-out infinite;transform-box:fill-box;transform-origin:50% 50%;animation-delay:@DLG@}
      .pbCloud{animation:pbDriftK @CLOUDT@s linear infinite}
      .pbWave{animation:pbWaveK @WAVET@s linear infinite}
      .pbWave2{animation:pbWaveK @WAVET2@s linear infinite}
      .pbLap{animation:pbLapK @LAPT@s ease-in-out infinite}
      .pbGull{animation:pbGullK @GULLT@s ease-in-out infinite}
      .pbZoom{animation:pbZoomK @ZOOMT@s linear infinite}
      .pbPulse{animation:pbPulseK @PULSET@s ease-in-out infinite;transform-box:fill-box;transform-origin:50% 50%}
      .pbSpray{animation:pbSprayK @SPRAYT@s ease-out infinite}
      .pbSfx{font-family:@FONT@;font-size:22px;font-weight:700;fill:@SFXC@;opacity:0.9}
      @keyframes pbSpinK{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
      @keyframes pbCrankK{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
      @keyframes pbBobK{0%,100%{transform:translateY(0)}50%{transform:translateY(-@BOB@px)}}
      @keyframes pbLegAK{0%,100%{transform:rotate(-@LEGA@deg)}50%{transform:rotate(@LEGA@deg)}}
      @keyframes pbLegBK{0%,100%{transform:rotate(@LEGA@deg)}50%{transform:rotate(-@LEGA@deg)}}
      @keyframes pbWingK{0%,100%{transform:rotate(@WGA@deg)}50%{transform:rotate(@WGB@deg)}}
      @keyframes pbDriftK{from{transform:translateX(0)}to{transform:translateX(150px)}}
      @keyframes pbWaveK{from{transform:translateX(0)}to{transform:translateX(-180px)}}
      @keyframes pbLapK{0%,100%{opacity:0.22;transform:translateX(0)}50%{opacity:0.85;transform:translateX(-26px)}}
      @keyframes pbGullK{0%{transform:translate(130px,0);opacity:0}25%{opacity:0.9}75%{opacity:0.9}100%{transform:translate(-230px,6px);opacity:0}}
      @keyframes pbZoomK{0%{transform:translateX(110px);opacity:0}35%{opacity:0.8}100%{transform:translateX(-190px);opacity:0}}
      @keyframes pbPulseK{0%,100%{opacity:0.55;transform:scale(1)}50%{opacity:0.95;transform:scale(1.07)}}
      @keyframes pbSprayK{0%{opacity:0.95;transform:translate(0,0) scale(1)}100%{opacity:0;transform:translate(@SPXD@px,@SPYD@px) scale(0.35)}}
    </style>
  </defs>

  <!-- 天空 / 太阳 / 云 / 海鸥 -->
  <rect x="0" y="0" width="720" height="440" fill="url(#pbSky)"/>
  <circle class="pbPulse" cx="@SUNX@" cy="@SUNY@" r="@SUNR@2" fill="url(#pbSunG)"/>
  <circle cx="@SUNX@" cy="@SUNY@" r="@SUNR@" fill="@SUN@"/>
@CLOUDS@
@GULLS@

  <!-- 海 + 浪 -->
  <rect x="0" y="240" width="720" height="104" fill="url(#pbSea)"/>
  <g class="pbWave"><path d="@WAVE1@" fill="none" stroke="@SEA1@" stroke-width="3.2" stroke-linecap="round" opacity="0.55"/></g>
  <g class="pbWave2"><path d="@WAVE2@" fill="none" stroke="@FOAM@" stroke-width="2.4" stroke-linecap="round" opacity="0.5"/></g>
  <path d="@LAP@" fill="none" stroke="@FOAM@" stroke-width="4" stroke-linecap="round" opacity="0.5" class="pbLap"/>

  <!-- 沙滩 -->
  <rect x="0" y="340" width="720" height="100" fill="url(#pbSand)"/>
  <ellipse cx="374" cy="345" rx="104" ry="9" fill="@SHADOW@" opacity="0.26"/>

  <!-- 速度线 / 水花 -->
@SPEED@
@SPRAY@

  <!-- 车 + 鹈鹕（整体轻微上下颠） -->
  <g class="pbBob">

    <!-- 后轮 -->
    <g transform="translate(@RWX@,@WY@)"><g class="pbSpinA">
      <circle cx="0" cy="0" r="@WR@" fill="none" stroke="#2c3242" stroke-width="6"/>
      <circle cx="0" cy="0" r="@WR2@" fill="none" stroke="#9aa4b2" stroke-width="3"/>
      @SPOKES1@
      <circle cx="0" cy="0" r="5" fill="#6b7280"/>
    </g></g>

    <!-- 前轮 -->
    <g transform="translate(@FWX@,@WY@)"><g class="pbSpinB">
      <circle cx="0" cy="0" r="@WR@" fill="none" stroke="#2c3242" stroke-width="6"/>
      <circle cx="0" cy="0" r="@WR2@" fill="none" stroke="#9aa4b2" stroke-width="3"/>
      @SPOKES2@
      <circle cx="0" cy="0" r="5" fill="#6b7280"/>
    </g></g>

    <!-- 车架 -->
    <g stroke="@FRAME@" stroke-width="5" stroke-linecap="round" fill="none">
      <path d="M@BBX@,@BBY@ L@SEATX@,@SEATY@"/>
      <path d="M@BBX@,@BBY@ L@HTX@,@HTY@"/>
      <path d="M@SEATX@,@SEATY@ L@HTX@,@HTY@"/>
      <path d="M@BBX@,@BBY@ L@RWX@,@WY@"/>
      <path d="M@SEATX@,@SEATY@ L@RWX@,@WY@"/>
      <path d="M@FWX@,@WY@ L@HTX@,@HTY@"/>
    </g>
    <path d="M330,246 q18,-9 33,2 q-15,7 -33,-2 z" fill="#2f3646"/>
    <path d="M@HTX@,@HTY@ q9,-13 22,-11" fill="none" stroke="#2f3646" stroke-width="5" stroke-linecap="round"/>
    <rect x="448" y="231" width="16" height="7" rx="3.5" fill="#1f2430" transform="rotate(-26 448 231)"/>

    <!-- 曲柄 + 踏板 + 牙盘 -->
    <g transform="translate(@BBX@,@BBY@)"><g class="pbCrank">
      <rect x="-30" y="-30" width="60" height="60" fill="none" stroke="none"/>
      <rect x="-4.5" y="-28" width="9" height="30" rx="4" fill="#4b5563"/>
      <rect x="-4.5" y="-2" width="9" height="30" rx="4" fill="#4b5563"/>
      <rect x="-9" y="-32" width="18" height="6" rx="2.6" fill="#1f2430"/>
      <rect x="-9" y="26" width="18" height="6" rx="2.6" fill="#1f2430"/>
      <circle cx="0" cy="0" r="14" fill="none" stroke="#9aa4b2" stroke-width="2.6"/>
      <circle cx="0" cy="0" r="4.6" fill="#9aa4b2"/>
    </g></g>

    <!-- 车筐（里面放鱼） -->
@BASKET@

    <!-- 后腿 / 前腿 -->
    <g transform="translate(@HIPX@,@HIPY@)"><g class="pbLegB">
      <path d="M0,0 L6,40 L2,78" fill="none" stroke="@LEGCOL@" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <ellipse cx="2" cy="78" rx="9" ry="4" fill="@LEGCOL@"/>
      <rect x="-45" y="-81" width="90" height="162" fill="none" stroke="none"/>
    </g></g>

    <!-- 身体 -->
    <ellipse cx="352" cy="224" rx="42" ry="29" fill="#ffffff"/>
    <ellipse cx="352" cy="233" rx="42" ry="19" fill="#e8eef8" opacity="0.55"/>
    <path d="M314,218 q-30,-7 -46,5 q22,11 46,6 z" fill="#f2f6fc"/>
    <path d="M300,224 q-24,-3 -36,6 q18,8 36,3 z" fill="#e4ebf6"/>

    <!-- 脖子 + 头 -->
    <path d="M382,214 C392,186 402,172 424,166" fill="none" stroke="#ffffff" stroke-width="26" stroke-linecap="round"/>
    <circle cx="424" cy="166" r="17" fill="#ffffff"/>
    <path d="M414,158 q-3,-15 11,-19 q-5,11 2,17 z" fill="@CREST@"/>

    <!-- 长喙 + 喉囊 -->
    <path d="M434,160 L524,176 L434,176 Z" fill="@BEAK@"/>
    <path d="M434,171 Q478,197 524,178 L434,178 Z" fill="@POUCH@" opacity="0.96"/>
    <path d="M436,163 L520,176" fill="none" stroke="#00000022" stroke-width="1.4"/>
    <circle cx="428" cy="161" r="3.7" fill="#1d2233"/>
    <circle cx="429.3" cy="159.8" r="1.2" fill="#ffffff"/>

    <!-- 翅膀（扇） -->
    <g transform="translate(348,212)"><g class="pbWing">
      <path d="M0,0 q27,-11 45,8 q-18,17 -45,8 q-10,-6 0,-16 z" fill="#eef3fa" stroke="#dde5f1" stroke-width="1.6"/>
      <path d="M6,4 q22,-6 34,6" fill="none" stroke="#dde5f1" stroke-width="1.4"/>
      <rect x="-52" y="-52" width="104" height="104" fill="none" stroke="none"/>
    </g></g>

    <!-- 挂件 -->
@ACC@
  </g>

  <!-- 拟声词 -->
@SFX@
  <!-- 天气 / 意境覆盖层(雪 / 雨 / 月夜 / 花瓣) -->
@WEATHER@
</svg>
"""


# ──────────────────────────────────────────────────────────────────────────────
# 意图判断
# ──────────────────────────────────────────────────────────────────────────────
def is_pelican_bike(text):
    """这句话是不是「给我画/写一个鹈鹕骑自行车」这类诉求。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(k in t for k in ("算法", "代码", "函数", "程序", "脚本", "报错", "原理", "源码", "实现一下")):
        return False
    if any(k in t for k in ("摩托", "电动车", "汽车", "卡车", "机车", "火车", "飞机", "轮椅", "滑板", "三轮")):
        return False
    bird = any(k in t for k in ("鹈鹕", "塘鹅", "pelican", "鹈"))
    if not bird:
        return False
    bike = any(k in t for k in ("自行车", "单车", "脚踏车", "骑行", "骑车", "蹬车", "猛猛蹬", "bike", "bicycle", "脚踏"))
    if bike:
        return True
    return any(k in t for k in ("画", "写", "来", "生成", "做个", "弄个", "搞个", "一张", "骑", "蹬"))


def is_pelican_draw(text):
    """只要是「画/写/来一张/做一只 鹈鹕…」就出图 —— 小方先想你的意思, 而不是只会骑自行车。
    顺带兼容旧的骑车关键词。给知识/代码/算法类问题让道, 别把"讲讲鹈鹕"误当成画图。"""
    if is_pelican_bike(text):
        return True
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(k in t for k in ("算法", "代码", "函数", "程序", "脚本", "报错", "原理", "源码", "实现一下", "是什么", "为什么", "怎么回答")):
        return False
    bird = any(k in t for k in ("鹈鹕", "塘鹅", "pelican", "鹈"))
    if not bird:
        return False
    if any(k in t for k in ("讲", "讲解", "科普", "知识", "能否", "可否")):
        if not any(k in t for k in ("画", "写", "来一张", "生成")):
            return False
    draw = any(k in t for k in ("画", "写", "来一张", "生成", "做一只", "弄个", "搞个", "要一只",
                                "来只", "画只", "画一幅", "画一个", "画鹈鹕", "一只鹈鹕", "个鹈鹕",
                                "只鹈鹕", "鹈鹕图", "鹈鹕画", "帮我画", "帮我写", "给我画", "给我写",
                                "画只鹈鹕", "拍一张"))
    return draw


# ──────────────────────────────────────────────────────────────────────────────
# 显式要求解析（v2.0）
#   以前画什么全凭随机，用户说"前面要有篮子"它有 28% 概率偏不画 —— 那就成抬杠了。
#   这里把用户这句里的硬要求抠出来，能抠到就照办，抠不到才交回随机（保持张张不同）。
# ──────────────────────────────────────────────────────────────────────────────
_CN_NUM = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

_NEG_KWS = ("不要", "不用", "别", "没有", "去掉", "拿掉", "不带", "不加",
            "无需", "不装", "不放", "免了", "省了", "不配", "没装")


def _cn_int(s):
    """把「3 / 三 / 十二 / 二十三」这种数词转成整数，转不动返回 None。"""
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    if s in _CN_NUM:
        return _CN_NUM[s]
    if len(s) == 2 and s[0] == "十" and s[1] in _CN_NUM:
        return 10 + _CN_NUM[s[1]]
    if len(s) == 2 and s[1] == "十" and s[0] in _CN_NUM:
        return _CN_NUM[s[0]] * 10
    if len(s) == 3 and s[1] == "十" and s[0] in _CN_NUM and s[2] in _CN_NUM:
        return _CN_NUM[s[0]] * 10 + _CN_NUM[s[2]]
    return None


def _negated_near(t, idx, span=7):
    """idx 之前 span 个字里有没有否定词 —— 用来区分「要有篮子」和「不要篮子」。"""
    lo = max(0, idx - span)
    return any(k in t[lo:idx] for k in _NEG_KWS)


def parse_constraints(text):
    """抽出这句里的显式要求。抽不到就留空，让画面自己随机去。"""
    t = (text or "").strip()
    cons = {}
    if not t:
        return cons

    # ① 车筐：要有 / 不要
    _bi = -1
    for w in ("篮子", "车筐", "筐", "basket"):
        _i = t.find(w)
        if _i >= 0:
            _bi = _i if _bi < 0 else min(_bi, _i)
    if _bi >= 0:
        cons["basket"] = not _negated_near(t, _bi)
        # ② 筐里几条鱼
        m = (re.search(r"([0-9零一二两三四五六七八九十]+)\s*(?:条|只|尾)?\s*鱼", t)
             or re.search(r"鱼[^0-9零一二两三四五六七八九十]{0,3}([0-9零一二两三四五六七八九十]+)\s*条", t))
        if m:
            n = _cn_int(m.group(1))
            if n is not None:
                cons["fish"] = max(1, min(6, n))
    elif re.search(r"([0-9零一二两三四五六七八九十]+)\s*(?:条|只|尾)\s*鱼", t):
        # 只说了装鱼、没说筐 —— 那鱼总得有地方放
        cons["basket"] = True
        n = _cn_int(re.search(r"([0-9零一二两三四五六七八九十]+)\s*(?:条|只|尾)\s*鱼", t).group(1))
        if n is not None:
            cons["fish"] = max(1, min(6, n))

    # ③ 配色：用户点名哪个就用哪个
    for p in _PB_PALETTES:
        if p["name"] in t:
            cons["palette"] = p["name"]
            break

    # ④ 挂件
    if any(k in t for k in ("素面", "不戴", "什么都别戴", "不挂", "没挂件", "不要挂件")):
        cons["acc"] = ""
    elif any(k in t for k in ("围巾", "围脖")):
        cons["acc"] = "scarf"
    elif any(k in t for k in ("墨镜", "眼镜")):
        cons["acc"] = "shades"
    elif any(k in t for k in ("小花", "朵花", "别朵花", "别着花", "头上戴花")):
        cons["acc"] = "flower"
    elif any(k in t for k in ("帽子", "凉帽", "小帽")):
        cons["acc"] = "cap"

    # ⑤ 指名要它喊的话
    m = re.search(r"(?:喊|叫|配音|来一声|吆喝)[一了声句出]{0,2}[「\"'“]?([^\s「」\"'“”，。！？]{2,8})[」\"'”]?", t)
    if m:
        cons["sfx"] = m.group(1)
    return cons


# ──────────────────────────────────────────────────────────────────────────────
# 生成
# ──────────────────────────────────────────────────────────────────────────────
def _spokes_path(n, r, phase):
    out = []
    for i in range(n):
        a = phase + i * (2.0 * math.pi / n)
        out.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#b6bfcc" stroke-width="1.5"/>'
                   % (r * math.cos(a), r * math.sin(a), -r * math.cos(a), -r * math.sin(a)))
    return "\n      ".join(out)


def _wave_path(y, up_down, humps, amp):
    d = ["M%d,%d q45,%d 90,0" % (-540, y, -amp * up_down)]
    d += ["t 90 0"] * (humps - 1)
    return " ".join(d)


def pelican_bike_svg(seed=None, info=None, cons=None, profile=None):
    """生成一段完整的动态 <svg> 文本。seed 传整数可复现，传 None 每次都不一样。
    cons: parse_constraints() 抽出来的显式要求（要/不要车筐、几条鱼、配色、挂件、喊什么）。
          给了 cons 就照办，没给的项继续随机 —— 所以既能听指令，也不会张张雷同。
    profile: pelican_intent() 出的「意境画像」(时间/天气/氛围)。
          有就切到匹配的配色并叠加氛围层(雪/雨/月夜/花瓣), 没有就维持原来的纯海边随机。"""
    rnd = random.Random(seed)
    cons = cons or {}
    pal = rnd.choice(_PB_PALETTES)
    _mood = (profile or {}).get("mood", "")
    if profile and profile.get("pal_override"):
        pal = profile["pal_override"]
    _want_pal = cons.get("palette")
    if _want_pal:
        for _p in _PB_PALETTES:
            if _p["name"] == _want_pal:
                pal = _p
                break
    spokes = rnd.choice((8, 10, 12))
    phase = rnd.uniform(0, math.pi)
    wheel_t = round(rnd.uniform(0.48, 0.9), 3)
    crank_t = round(wheel_t * rnd.uniform(1.6, 2.2), 3)
    leg_a = rnd.randint(9, 15)
    wing_a = rnd.randint(-11, -2)
    wing_b = rnd.randint(6, 17)
    wing_t = round(rnd.uniform(0.7, 1.5), 3)
    bob = rnd.randint(2, 5)
    bob_t = round(rnd.uniform(0.85, 1.5), 3)
    sunx, suny, sunr = rnd.randint(95, 620), rnd.randint(58, 150), rnd.randint(26, 42)
    wave_t1 = round(rnd.uniform(4.5, 7.5), 3)
    wave_t2 = round(rnd.uniform(6.5, 10.5), 3)
    lap_t = round(rnd.uniform(3.2, 5.4), 3)
    cloud_t = round(rnd.uniform(26, 46), 3)
    gull_t = round(rnd.uniform(32, 58), 3)
    zoom_t = round(rnd.uniform(0.7, 1.3), 3)
    pulse_t = round(rnd.uniform(3.4, 6.2), 3)
    spray_t = round(rnd.uniform(0.65, 1.15), 3)
    spxd, spyd = rnd.randint(-64, -32), rnd.randint(-42, -16)
    acc = rnd.choice(("", "cap", "cap", "scarf", "shades", "flower"))
    acc_col = rnd.choice(("#e4574f", "#f0a63c", "#7cc4e8", "#a97fe0", "#5ec98f", "#ff8fb1"))
    leg_col = rnd.choice(("#d99b3f", "#e0a944", "#c98a33"))
    beak_col = rnd.choice(("#f0a63c", "#e8a13a", "#f2b25a"))
    pouch_col = rnd.choice(("#f7c46c", "#f2b95e", "#f9cd80"))
    crest_col = rnd.choice(("#ffd7a1", "#ffe0b3", "#f6c98c"))
    frame_col = rnd.choice(("#2c3242", "#3c4a63", "#7a3f46", "#2f5d50"))
    basket_col = rnd.choice(("#e3c184", "#d9b271", "#efd39a"))
    fishn = rnd.randint(1, 3)
    cloudn = rnd.randint(2, 4)
    gulln = rnd.randint(0, 3)
    sprayn = rnd.randint(3, 6)
    has_basket = rnd.random() < 0.72
    sfx = rnd.choice(_PB_SFX) if rnd.random() < 0.62 else ""

    # ── 用户点名要求优先于随机 ───────────────────────────────────────────
    _honored = []
    if _mood:
        _honored.append("意境 → " + _mood)
    if cons.get("basket") is not None:
        has_basket = bool(cons["basket"])
        _honored.append("车筐 → " + ("要" if has_basket else "不要"))
    if has_basket:
        if cons.get("fish") is not None:
            fishn = int(cons["fish"])
            _honored.append("筐里 %d 条鱼" % fishn)
    else:
        fishn = 0
    if cons.get("palette"):
        _honored.append("配色 → " + cons["palette"])
    if cons.get("acc") is not None:
        acc = cons["acc"]
        _honored.append("挂件 → " + (_PB_ACC_DESC.get(acc, "") or "什么也没戴"))
    if cons.get("sfx"):
        sfx = cons["sfx"]
        _honored.append("喊 → " + sfx)

    # 云
    clouds = []
    for i in range(cloudn):
        cx, cy = rnd.randint(20, 690), rnd.randint(34, 132)
        sc = round(rnd.uniform(0.6, 1.25), 2)
        op = round(rnd.uniform(0.5, 0.92), 2)
        dl = "-%.2fs" % rnd.uniform(0, cloud_t)
        clouds.append(
            '<g class="pbCloud" style="animation-delay:%s%s">'
            '<g transform="translate(%d,%d) scale(%s)" opacity="%s">'
            '<ellipse cx="0" cy="0" rx="34" ry="15" fill="#ffffff"/>'
            '<ellipse cx="24" cy="5" rx="26" ry="12" fill="#ffffff"/>'
            '<ellipse cx="-22" cy="6" rx="22" ry="10" fill="#ffffff"/></g></g>'
            % (dl, "", cx, cy, sc, op))

    # 海鸥
    gulls = []
    for i in range(gulln):
        gy = rnd.randint(66, 176)
        sc = round(rnd.uniform(0.7, 1.15), 2)
        dl = "-%.2fs" % rnd.uniform(0, gull_t)
        gulls.append(
            '<g class="pbGull" style="animation-delay:%s">'
            '<g transform="translate(%d,%d) scale(%s)" fill="none" stroke="#ffffff" stroke-width="2.4" '
            'stroke-linecap="round" opacity="0.85">'
            '<path d="M-13,0 q7,-9 13,0 q6,-9 13,0"/></g></g>'
            % (dl, rnd.randint(180, 520), gy, sc))

    # 速度线
    speed = []
    for i in range(4):
        y = rnd.randint(276, 336)
        ln = rnd.randint(38, 92)
        x0 = rnd.randint(60, 300)
        dl = "-%.2fs" % rnd.uniform(0, zoom_t)
        op = round(rnd.uniform(0.25, 0.55), 2)
        speed.append('<line class="pbZoom" style="animation-delay:%s" x1="%d" y1="%d" x2="%d" y2="%d" '
                     'stroke="#ffffff" stroke-width="2.6" stroke-linecap="round" opacity="%s"/>'
                     % (dl, x0, y, x0 - ln, y, op))

    # 水花
    spray = []
    for i in range(sprayn):
        cx = rnd.randint(478, 560)
        cy = rnd.randint(330, 340)
        r = round(rnd.uniform(2.2, 4.6), 2)
        dl = "-%.2fs" % rnd.uniform(0, spray_t)
        spray.append('<circle class="pbSpray" style="animation-delay:%s" cx="%d" cy="%d" r="%s" fill="%s" opacity="0.9"/>'
                     % (dl, cx, cy, r, pal["sea"][0]))

    # 辐条
    sp1 = _spokes_path(spokes, _WR - 5, phase)
    sp2 = _spokes_path(spokes, _WR - 5, phase + math.pi / spokes)

    # 车筐 + 鱼
    if has_basket:
        fish = []
        for i in range(fishn):
            fx = 444 + i * 11 + rnd.randint(-1, 2)
            fy = rnd.randint(203, 213)
            fc = rnd.choice(("#8fd3ff", "#b8e6f7", "#9ad0e8", "#ffd6a5"))
            fish.append('<ellipse cx="%d" cy="%d" rx="6.6" ry="3.4" fill="%s"/>'
                        '<path d="M%d,%d l-7,-4 l0,8 z" fill="%s"/>' % (fx, fy, fc, fx - 6, fy, fc))
        basket = ('    <g>\n      <path d="M436,214 l42,0 l-6,26 l-30,0 z" fill="%s" stroke="#b9932f" stroke-width="1.6"/>\n'
                  '      <path d="M440,221 l34,0 M441,228 l32,0" stroke="#b9932f" stroke-width="1.2" fill="none"/>\n'
                  '      %s\n    </g>' % (basket_col, "\n      ".join(fish)))
    else:
        basket = "    <!-- 这次没装车筐 -->"

    # 挂件
    acc_map = {
        "cap": '<path d="M410,152 q15,-13 31,-4 l2,6 l-33,2 z" fill="%s"/>'
               '<path d="M441,154 l17,4 l-19,2 z" fill="%s"/>' % (acc_col, acc_col),
        "scarf": '<path d="M404,194 q-14,3 -21,15 q11,5 18,-5 z" fill="%s"/>'
                 '<path d="M386,206 q-12,10 -14,26 q7,-2 10,-12 z" fill="%s"/>' % (acc_col, acc_col),
        "shades": '<path d="M416,157 l20,3 l-2,8 l-18,-2 z" fill="#243043" opacity="0.92"/>',
        "flower": '<circle cx="416" cy="151" r="4.4" fill="%s"/><circle cx="416" cy="151" r="1.7" fill="#fff5c2"/>' % acc_col,
    }
    acc_svg = "    " + acc_map.get(acc, "<!-- 这次素面朝天 -->")

    sfx_svg = ""
    if sfx:
        sfx_svg = ('  <text class="pbSfx" x="%d" y="%d" transform="rotate(-8 %d %d)">%s</text>'
                   % (rnd.randint(486, 560), rnd.randint(170, 232), 520, 200, sfx))

    svg = _PB_SVG_TPL
    rep = {
        "@SKY1@": pal["sky"][0], "@SKY2@": pal["sky"][1], "@SKY3@": pal["sky"][2],
        "@SEA1@": pal["sea"][0], "@SEA2@": pal["sea"][1],
        "@SAND1@": pal["sand"][0], "@SAND2@": pal["sand"][1],
        "@SUN@": pal["sun"], "@SUNCORE@": pal["core"], "@SHADOW@": pal["shadow"],
        "@FOAM@": "#ffffff",
        "@SUNX@": str(sunx), "@SUNY@": str(suny), "@SUNR@": str(sunr),
        "@BOB@": str(bob), "@BOBT@": str(bob_t),
        "@WHEELT@": str(wheel_t), "@CRANKT@": str(crank_t),
        "@LEGA@": str(leg_a), "@WGA@": str(wing_a), "@WGB@": str(wing_b), "@WINGT@": str(wing_t),
        "@CLOUDT@": str(cloud_t), "@WAVET@": str(wave_t1), "@WAVET2@": str(wave_t2),
        "@LAPT@": str(lap_t), "@GULLT@": str(gull_t), "@ZOOMT@": str(zoom_t),
        "@PULSET@": str(pulse_t), "@SPRAYT@": str(spray_t),
        "@SPXD@": str(spxd), "@SPYD@": str(spyd),
        "@DLW@": "-%.2fs" % rnd.uniform(0, wheel_t), "@DLC@": "-%.2fs" % rnd.uniform(0, crank_t),
        "@DLB@": "-%.2fs" % rnd.uniform(0, bob_t), "@DLG@": "-%.2fs" % rnd.uniform(0, wing_t),
        "@WAVE1@": _wave_path(246, 1, 17, 9), "@WAVE2@": _wave_path(286, -1, 17, 7),
        "@LAP@": _wave_path(338, 1, 17, 4),
        "@RWX@": str(_RWX), "@WY@": str(_WY), "@WR@": str(_WR), "@WR2@": str(_WR - 8),
        "@FWX@": str(_FWX), "@BBX@": str(_BBX), "@BBY@": str(_BBY),
        "@SEATX@": "348", "@SEATY@": "252", "@HTX@": "438", "@HTY@": "248",
        "@HIPX@": str(_HIPX), "@HIPY@": str(_HIPY),
        "@SPOKES1@": sp1, "@SPOKES2@": sp2,
        "@FRAME@": frame_col, "@LEGCOL@": leg_col, "@BEAK@": beak_col,
        "@POUCH@": pouch_col, "@CREST@": crest_col,
        "@CLOUDS@": "\n  ".join(clouds), "@GULLS@": "\n  ".join(gulls),
        "@SPEED@": "\n  ".join(speed), "@SPRAY@": "\n  ".join(spray),
        "@BASKET@": basket, "@ACC@": acc_svg, "@SFX@": sfx_svg, "@FONT@": _FONT,
        "@SFXC@": "#fff8e7" if pal["name"] != "正午" else "#3b3b3b",
        "@WEATHER@": _weather_overlay(profile) if profile else "",
    }
    for k, v in rep.items():
        svg = svg.replace(k, v)

    if info is not None:
        info.update({
            "seed": seed,
            "palette": pal["name"],
            "spokes": spokes,
            "acc": acc,
            "acc_desc": _PB_ACC_DESC.get(acc, ""),
            "fish": fishn if has_basket else 0,
            "basket": bool(has_basket),
            "cadence": crank_t,
            "wheel_t": wheel_t,
            "clouds": cloudn,
            "gulls": gulln,
            "sfx": sfx,
            "frame": frame_col,
            "cons": dict(cons),
            "honored": list(_honored),
            "mood": _mood,
            "variant": "%s · %d 根辐条 · %s · 车筐 %d 条鱼 · 踏频约 %.1f 秒一圈"
                       % (pal["name"], spokes, _PB_ACC_DESC.get(acc, ""), fishn if has_basket else 0, crank_t),
        })
    return svg


def pelican_bike_save(svg, info=None, folder=None):
    """把 SVG 落盘。返回绝对路径；出任何问题都返回空串，绝不抛异常打断对话。"""
    try:
        base = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), "产物")
        os.makedirs(base, exist_ok=True)
        now = datetime.datetime.now()
        pal = (info or {}).get("palette", "画面")
        name = "鹈鹕骑自行车_%s_%s.svg" % (pal, now.strftime("%m%d_%H%M%S"))
        path = os.path.join(base, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        if info is not None:
            info["path"] = path
            info["filename"] = name
            try:
                info["rel"] = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
            except Exception:
                info["rel"] = name
        return path
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# v2.8 · 鹈鹕引擎「先想意图再出图」
#   用户原话:"代价是要认真写一下的, 不要直接套模板。运作改成图像生成 … 还是能生成
#   鹈鹕, 但需要让小方好好思考一下用户的意图, 然后再输出。"
#   落地(用户选的方案): 本地即时出 SVG(非模板, 按意图换氛围) + 小方写好的真图 Prompt。
#   三层: ① pelican_intent 读意图 → ② 景象覆盖层(_weather_overlay)切氛围配色+动画
#         → ③ pelican_reply 编排"先想的证据 + 画成啥 + 真图 Prompt 落盘 txt"。
# ══════════════════════════════════════════════════════════════════════════════

# 意境专属配色(平时《_PB_PALETTES》里没有的: 雪 / 雨 / 月夜 / 草原花海)
_WEATHER_PALS = {
    "snow": {"name": "雪景", "sky": ("#dfe9f3", "#f3f7fb", "#ffffff"), "sea": ("#7f98b8", "#4d678a"),
             "sand": ("#eef2f6", "#d9e2ec"), "sun": "#f4f8ff", "core": "#ffffff", "shadow": "#9aa9bd"},
    "rain": {"name": "雨天", "sky": ("#8fa4ba", "#c0cad6", "#dde4ec"), "sea": ("#4c6a86", "#2a4058"),
             "sand": ("#ccd4dd", "#aab6c2"), "sun": "#cfd8e2", "core": "#edf2f6", "shadow": "#67707e"},
    "moon": {"name": "月夜", "sky": ("#10173a", "#1d2a5e", "#3f4f94"), "sea": ("#1a2a55", "#0b1430"),
             "sand": ("#5a5c7c", "#3a3c58"), "sun": "#f2f5ff", "core": "#ffffff", "shadow": "#181b30"},
    "petals": {"name": "花海", "sky": ("#b9e2ff", "#e3f4ff", "#f6fbea"), "sea": ("#3a7a5e", "#1e4a39"),
               "sand": ("#e3efc8", "#c6daa0"), "sun": "#ffd98a", "core": "#fff3c4", "shadow": "#7a9466"},
}


def _weather_overlay(profile):
    """把意境渲染成一段叠加在整张画最上层的自包含 SVG 层(雪 / 雨 / 月夜 / 花瓣)。
    内部自带 <style> 动画, 不碰主模板结构 —— 加不坏, 空意境就返回空串。"""
    try:
        _wet = (profile or {}).get("weather", "") or ""
        if not _wet:
            return ""
        _seed = (profile or {}).get("seed", None)
        rnd = random.Random(_seed)
        parts = ['<g class="pbWeather" style="pointer-events:none"><style>']
        if _wet == "snow":
            parts.append("@keyframes pbFall{from{transform:translateY(-40px)}to{transform:translateY(500px)}}")
            parts.append("</style>")
            for i in range(22):
                parts.append('<circle class="pbSnow" cx="%d" cy="%d" r="%s" '
                             'style="fill:#fff;opacity:.92;animation:pbFall %ss linear infinite;animation-delay:-%ss"/>'
                             % (rnd.randint(6, 714), rnd.randint(-60, 440),
                                round(rnd.uniform(2, 4.6), 1), round(rnd.uniform(2.4, 5.6), 1),
                                round(rnd.uniform(0, 5.0), 2)))
        elif _wet == "rain":
            parts.append("@keyframes pbDrift{from{transform:translate(0,-30px)}to{transform:translate(-60px,520px)}}")
            parts.append("</style>")
            for i in range(30):
                x, y, ln = rnd.randint(4, 712), rnd.randint(-60, 300), round(rnd.uniform(18, 30), 1)
                parts.append('<line class="pbRain" x1="%d" y1="%d" x2="%d" y2="%d" '
                             'style="stroke:#b7d2ef;stroke-width:2;opacity:.6;animation:pbDrift %ss linear infinite;animation-delay:-%ss"/>'
                             % (x, y, x + 8, y + ln, round(rnd.uniform(.7, 1.6), 2), round(rnd.uniform(0, 2.2), 2)))
        elif _wet == "moon":
            parts.append("@keyframes pbTw{0%{opacity:.15}50%{opacity:1}100%{opacity:.15}}")
            parts.append("</style>")
            for i in range(18):
                parts.append('<circle class="pbStar" cx="%d" cy="%d" r="%s" '
                             'style="fill:#fff;animation:pbTw %ss ease-in-out infinite;animation-delay:-%ss"/>'
                             % (rnd.randint(30, 690), rnd.randint(20, 120), round(rnd.uniform(1.1, 2.3), 1),
                                round(rnd.uniform(1.6, 3.4), 2), round(rnd.uniform(0, 3.0), 2)))
            parts.append('<circle cx="648" cy="64" r="44" fill="#f2f5ff" opacity="0.96"/>')
            parts.append('<circle cx="630" cy="52" r="24" fill="#181f42" opacity="0.9"/>')
            parts.append('<circle cx="660" cy="70" r="18" fill="#181f42" opacity="0.85"/>')
        elif _wet == "petals":
            parts.append("@keyframes pbDrift2{from{transform:translate(0,-20px) rotate(0)}"
                         "to{transform:translate(70px,520px) rotate(300deg)}}")
            parts.append("</style>")
            for i in range(26):
                parts.append('<ellipse class="pbPet" cx="%d" cy="%d" rx="6.5" ry="3.6" '
                             'style="fill:%s;opacity:.85;animation:pbDrift2 %ss linear infinite;animation-delay:-%ss"/>'
                             % (rnd.randint(6, 700), rnd.randint(-60, 300),
                                rnd.choice(("#ffb7cf", "#ffcfe0", "#ff9dbd", "#ffe3ee")),
                                round(rnd.uniform(3.6, 7.5), 1), round(rnd.uniform(0, 6.0), 2)))
        else:
            return ""
        parts.append('</g>')
        return "\n  ".join(parts)
    except Exception:
        return ""


def pelican_intent(text, rnd=None):
    """小方在动笔前先「读你的意思」: 抽出时间 / 天气 / 氛围, 凝成一句中文画面 + 一条英文真图 Prompt。
    返回 {"theme", "weather", "mood", "desc_cn", "prompt_en", "pal_override", "seed"}。
    抽不到就回沿海积心态(theme=rnd.choice(第一阶段)), Prompt 仍由这轮随机打磨 —— 每次不重样。"""
    rnd = rnd or random
    t = (text or "").strip()
    # ① 时间 → 配色
    _pal = None
    if any(k in t for k in ("清晨", "黎明", "早上", "早晨")):
        _pal = "清晨"
    elif any(k in t for k in ("正午", "中午", "午后", "大晴天")):
        _pal = "正午"
    elif any(k in t for k in ("黄昏", "日落", "夕阳", "晚霞")):
        _pal = "黄昏"
    elif any(k in t for k in ("傍晚",)):
        _pal = "傍晚"
    elif any(k in t for k in ("深夜", "午夜", "夜晚", "晚上", "夜")):
        _pal = "夜色"
    # ② 天气 → 氛围覆盖层 + 专属配色(优先级高于时间)
    _wet = ""
    if any(k in t for k in ("雪", "飘雪", "下雪", "雪地", "雪花")):
        _wet = "snow"
    elif any(k in t for k in ("下雨", "雨天", "落雨", "雨")):
        _wet = "rain"
    elif any(k in t for k in ("月亮", "月光", "月下", "月色", "星空", "星星")):
        _wet = "moon"
    elif any(k in t for k in ("落花", "花瓣", "樱花", "繁花", "花雨")):
        _wet = "petals"
    # ③ 场景情绪词
    _mood = "海边骑行"
    for _kw, _w in (("圣诞", "圣诞暖光里慢骑"), ("雪", "雪中慢行"), ("雨", "雨里踩水"),
                    ("月亮", "月下夜骑"), ("月光", "月下夜骑"), ("夕阳", "追着夕阳"),
                    ("清晨", "晨光里轻骑"), ("草原", "草地撒欢"), ("花", "花间穿行"),
                    ("风", "逆风飞驰"), ("山顶", "山顶迎风"), ("桥", "跨海大桥")) :
        if _kw in t:
            _mood = _w
            break
    # ④ 配色取舍: 天气 > 时间 > 无(维持海边随机)
    _po = None
    if _wet in _WEATHER_PALS:
        _po = _WEATHER_PALS[_wet]
    elif _pal and _pal in ("清晨", "正午", "黄昏", "夜色"):
        for _p in _PB_PALETTES:
            if _p["name"] == _pal:
                _po = _p
                break
    # ⑤ 中文一句话: 小方眼里 "这幅画长什么样"
    _if_cn = {
        "snow": "漫天雪花正轻轻飘落", "rain": "细雨里车轮碾过水洼",
        "moon": "一轮明月悬在海面上", "petals": "樱花花瓣随风洒落",
    }.get(_wet, "")
    if not _if_cn:
        _if_cn = {
            "清晨": "清晨柔光铺开", "正午": "正午艳阳高照", "黄昏": "夕阳把天际烧成橙红",
            "夜色": "入夜, 海面倒映月光",
        }.get(_pal, "海风正劲, 浪花一层层翻涌")
    _theme = _wet or _pal or "海边"
    _desc = "%s —— %s, 一只鹈鹕正猛蹬着自行车在海边飞驰。" % (_if_cn, _mood)
    # ⑥ 英文真图 Prompt: 认到意图 → 加上对应光线/天气 → 随机打磨画风
    _acc = ""
    if any(k in t for k in ("围巾", "围脖")):
        _acc = ", wearing a cozy scarf"
    elif any(k in t for k in ("墨镜", "眼镜")):
        _acc = ", wearing cool sunglasses"
    elif any(k in t for k in ("帽子", "凉帽")):
        _acc = ", wearing a little cap"
    elif any(k in t for k in ("朵花", "别朵花", "戴花")):
        _acc = ", with a small flower on its head"
    _light = {
        "snow": "falling snow, soft cold winter light", "rain": "light rainfall, cool blue-grey tones",
        "moon": "bright moonlight and stars above the sea at night", "petals": "cherry blossoms drifting, gentle daylight",
    }.get(_wet, {
        "清晨": "soft golden morning glow", "正午": "bright midday sunshine",
        "黄昏": "warm golden sunset over the sea", "夜色": "calm night sky by the sea",
    }.get(_pal, "sunny coastal breeze"))
    _style = rnd.choice((
        "flat vector illustration, clean pastel palette, soft shading",
        "charming storybook illustration, warm colors",
        "minimal cute style, rounded shapes, lively",
        "semi-realistic soft render, gentle gradients"))
    _prompt = ("a cute white pelican%s, furiously cycling a vintage bicycle along the seashore "
               "under %s, %s, high quality, 4k, crisp edges") % (_acc, _light, _style)
    return {"theme": _theme, "weather": _wet, "mood": _mood, "desc_cn": _desc,
            "prompt_en": _prompt, "pal_override": _po}


def pelican_save_prompt(prompt, info=None, folder=None):
    """把"小方写好的真图 Prompt"落盘成 .prompt.txt(跟 SVG 同目录同名), 返回绝对路径。"""
    try:
        base = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), "产物")
        os.makedirs(base, exist_ok=True)
        stem = ((info or {}).get("filename", "鹈鹕") or "鹈鹕").rsplit(".", 1)[0]
        path = os.path.join(base, stem + ".prompt.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt if prompt.endswith("\n") else prompt + "\n")
        return path
    except Exception:
        return ""


def pelican_reply(text="鹈鹕骑自行车", seed=None, save=True, folder=None, cons=None):
    """依你的意图出「这只鹈鹕」: 小方先想(读意图) → 生成动态 SVG → 落盘 SVG + 真图 Prompt。
    返回一段可以直接回给用户的文本(含小方怎么想的 + 画成啥 + 真图 Prompt 存哪)。"""
    rnd = random.Random(seed) if seed is not None else random
    info = {}
    if cons is None:
        cons = parse_constraints(text)
    profile = pelican_intent(text, rnd)
    profile["seed"] = seed if seed is not None else rnd.random()
    svg = pelican_bike_svg(seed, info, cons, profile)
    path = pelican_bike_save(svg, info, folder) if save else ""
    prompt = profile.get("prompt_en", "") or ""
    ppath = ""
    if save and prompt:
        _head = "小方为「这只鹈鹕」写好的真图 Prompt(可直接复制喂给任意 AI 画图):\n\n"
        ppath = pelican_save_prompt(_head + prompt, info, folder)
    lines = []
    lines.append("🎨 小方先想了下你的意思: " + profile.get("desc_cn", ""))
    lines.append("")
    lines.append("《鹈鹕 · %s》这次画的是:" % (profile.get("theme") or info["palette"]))
    lines.append("- 配色: " + info["palette"])
    lines.append("- 辐条: %d 根" % info["spokes"])
    lines.append("- 挂件: " + (info["acc_desc"] or "什么也没戴"))
    lines.append("- 车筐: " + ("%d 条鱼" % info["fish"] if info["basket"] else "这次没装"))
    lines.append("- 踏频: 约 %.1f 秒一圈" % info["cadence"])
    if info.get("mood"):
        lines.append("- 意境: " + info["mood"])
    if info.get("honored"):
        lines.append("- 按你点名的: " + "；".join(info["honored"]))
    if path or ppath:
        lines.append("")
        lines.append("存成了:")
        if path:
            lines.append("  · 动态 SVG → " + info.get("rel", path))
        if ppath:
            try:
                _rl = os.path.relpath(ppath, os.path.dirname(os.path.abspath(__file__)))
            except Exception:
                _rl = ppath
            lines.append("  · 真图 Prompt → " + _rl)
    if prompt:
        lines.append("")
        lines.append("✍️ 小方写好的真图 Prompt(复制喂给任何 AI 画图, 就是这只鹈鹕):")
        lines.append(prompt)
    if not path:
        lines.append("")
        lines.append("（落盘没成功, 但画面已经生成好了）")
    return "\n".join(lines)


def pelican_bike_reply(text="鹈鹕骑自行车", seed=None, save=True, folder=None, cons=None):
    """一步到位：解析要求 + 随机生成 + 落盘 + 组装一段可以直接回给用户的文本。"""
    info = {}
    if cons is None:
        cons = parse_constraints(text)
    svg = pelican_bike_svg(seed, info, cons)
    path = pelican_bike_save(svg, info, folder) if save else ""
    lines = [
        "《鹈鹕骑自行车》· 这次画的是:",
        "- 配色: " + info["palette"],
        "- 辐条: %d 根" % info["spokes"],
        "- 挂件: " + (info["acc_desc"] or "什么也没戴"),
        "- 车筐: " + ("%d 条鱼" % info["fish"] if info["basket"] else "这次没装"),
        "- 踏频: 约 %.1f 秒一圈" % info["cadence"],
    ]
    if info.get("sfx"):
        lines.append("- 它喊了一声: " + info["sfx"])
    if info.get("honored"):
        lines.append("- 按你点名要求改的: " + "；".join(info["honored"]))
    if path:
        lines += ["", "存成了: " + info.get("rel", path), "用浏览器打开就是动的 —— 轮子在转, 腿在蹬, 翅膀在扇, 浪在推。"]
    else:
        lines += ["", "（落盘没成功, 但画面已经生成好了）"]
    return "\n".join(lines)


# ── 鹈鹕引擎的对外收口(内联后依旧是"一个统一入口") ──────────────────────────
class _PelicanInline(object):
    """兼容壳: 以前 PELICAN / _get_pelican() 给的是模块对象, 内联后给一个
    同名同接口的对象, 老调用点一行都不用改。"""

    is_pelican_bike = staticmethod(is_pelican_bike)
    is_pelican_draw = staticmethod(is_pelican_draw)
    parse_constraints = staticmethod(parse_constraints)
    pelican_bike_svg = staticmethod(pelican_bike_svg)
    pelican_bike_save = staticmethod(pelican_bike_save)
    pelican_bike_reply = staticmethod(pelican_bike_reply)
    pelican_intent = staticmethod(pelican_intent)
    pelican_save_prompt = staticmethod(pelican_save_prompt)
    pelican_reply = staticmethod(pelican_reply)


PELICAN = _PelicanInline()              # 内联后恒可用, 不再是"可能为 None 的占位"


def _get_pelican():
    """内联版: 引擎就在本文件里, 恒可用, 不会再出现"少一个文件就开不了机"。"""
    return PELICAN


PELICAN_SVG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "产物")


def pelican_answer(text):
    """兼容旧入口(名字沿旧)—— 现为通用 SVG/CSS 图像引擎入口, 不再只画鹈鹕。
    是「画/写/生成 XXX + 气氛」就出图; 讲知识/写代码/问算法给 None, 让位别的路由。"""
    return svgdream_answer(text)


# ══════════════════════════════════════════════════════════════════════════════
# V2.8 · 通用 SVG/CSS 图像生成层
#   鹈鹕只是它可以画的主题之一。这里把「主体识别 + 气氛意图 + 通用画布 + 主体几何」
#   接到一起: 你说画什么、什么气氛, 引擎就用几何部件 + CSS 动画现场画出来, 并写一条
#   真正可用的真图 Prompt。全部纯标准库, import 本文件不加载任何神经网络/不占显存。
# ══════════════════════════════════════════════════════════════════════════════

SVGDREAM_SVG_DIR = PELICAN_SVG_DIR

# ══════════════════════════════════════════════════════════════════════════════
# V2.9 · 联想词库 + 主体库 + 可插拔引擎注册表
#   「学精」的三件事:
#   ① 词是连成网的, 不是孤立的触发词 —— 名词/动词/形容词/场景都连到主体, 还带权重。
#   ② 主体大幅扩充(35+ 类), 分组挂到不同引擎; 引擎可插拔, 未来新引擎 import 后
#      register_engine() 自动挂载, 各管一撮主体, 互不打架。
#   ③ 模糊意图: 用户没说清画啥, 只给了动词/形容词/场景, 也能靠联想投票推断出主体。
# ══════════════════════════════════════════════════════════════════════════════

#  词条类型:
#    n=名词直呼   v=动词(动作联想)   a=形容词(属性联想)   s=场景(环境联想)
#   权重只是「联想亲缘度」, 名词最亲, 多词命中叠加重 —— 所以越隐晦越靠其它线索补。
_CONNECT = [
    # 动词: 动作 → 能做这动作 / 常被这么形容的主体
    ("v", "游",       {"fish": 3, "whale": 3, "penguin": 2, "dolphin": 2, "boat": 2, "submarine": 1}),
    ("v", "遨游",     {"fish": 3, "whale": 3, "dolphin": 3}),
    ("v", "翱翔",     {"bird": 3, "rocket": 2}),
    ("v", "飞翔",     {"bird": 4, "dragon": 2}),
    ("v", "飞",       {"bird": 2, "dragon": 2, "rocket": 1}),
    ("v", "喷水",     {"whale": 4}),
    ("v", "跃出",     {"fish": 3, "whale": 3}),
    ("v", "跃",       {"dolphin": 3, "fish": 3}),
    ("v", "窜",       {"rocket": 2}),
    ("v", "奔跑",     {"horse": 3, "dog": 2, "cat": 1, "fox": 2}),
    ("v", "奔腾",     {"horse": 4}),
    ("v", "撒欢",     {"dog": 3, "hamster": 1}),
    ("v", "打滚",     {"cat": 2, "dog": 3}),
    ("v", "摇尾巴",   {"dog": 3, "cat": 2}),
    ("v", "追",       {"cat": 2, "dog": 3}),
    ("v", "发光",     {"lighthouse": 2, "moon": 2, "sun": 2, "star": 2}),
    ("v", "升起",     {"sun": 3, "moon": 2}),
    ("v", "降落",     {"ufo": 3}),
    ("v", "起飞",     {"rocket": 3, "bird": 2}),
    ("v", "升空",     {"rocket": 3}),
    ("v", "喷火",     {"volcano": 4, "dragon": 3, "rocket": 2}),
    ("v", "发芽",     {"tree": 3}),
    ("v", "开花",     {"tree": 3}),
    ("v", "咩",       {"sheep": 3}),
    ("v", "蹲坐",     {"cat": 2, "frog": 2}),
    ("v", "呱",       {"frog": 3}),
    ("v", "哞",       {"cow": 3}),
    ("v", "漫步",     {"penguin": 2}),
    ("v", "摇摆",     {"penguin": 3, "boat": 1}),
    ("v", "吐泡",     {"fish": 2, "whale": 1}),
    ("v", "拉蘑菇",   {"cloud": 2}),
    # 形容词/属性: 特征 → 最配的主体
    ("a", "雪白",     {"snowman": 3, "rabbit": 2, "cloud": 2}),
    ("a", "圆滚滚",   {"hamster": 2, "cat": 2, "penguin": 2}),
    ("a", "毛茸茸",   {"cat": 3, "bear": 3, "rabbit": 2, "dog": 2}),
    ("a", "软乎乎",   {"cat": 2, "rabbit": 2, "cloud": 2}),
    ("a", "五颜六色", {"rainbow": 3, "fish": 2}),
    ("a", "迷路",     {"cat": 2}),
    ("a", "刺眼",     {"sun": 2}),
    ("a", "巨大",     {"whale": 2}),
    ("a", "迷你",     {"hamster": 2}),
    ("a", "健壮",     {"horse": 2}),
    ("a", "神秘",     {"ufo": 2, "moon": 1}),
    ("a", "巍峨",     {"mountain": 2}),
    ("a", "朦胧",     {"mountain": 2, "moon": 1}),
    # 场景/环境: 场地 → 常出现的它
    ("s", "海边",     {"lighthouse": 3, "boat": 2, "pelican": 3, "whale": 2}),
    ("s", "沙滩",     {"crab": 3, "pelican": 2}),
    ("s", "天上",     {"moon": 2, "star": 2, "sun": 2, "ufo": 2, "cloud": 3}),
    ("s", "太空",     {"rocket": 3, "planet": 3, "star": 2, "ufo": 2}),
    ("s", "星空下",   {"moon": 2, "star": 2}),
    ("s", "草原",     {"horse": 3, "sheep": 2, "cow": 2}),
    ("s", "田",       {"cow": 2, "horse": 1}),
    ("s", "海底",     {"fish": 3, "whale": 2, "submarine": 2}),
    ("s", "火山口",   {"volcano": 3}),
    ("s", "城堡前",   {"castle": 1}),
]


#  主体库: kind → (中文名, 英文真图描述, 触发别名清单)
#  别名(n)在词表里自动给该 kind 加满权重(命中即坐实); 其余词靠 _CONNECT 联想加分。
_ENTITIES = {}
def _ent(kind, cn, en, defs):
    """登记一个可画主体: 别名直接进词典, 让「直呼主体」必命中。"""
    _ENTITIES[kind] = {"cn": cn, "en": en, "defs": defs}
    for w in defs:
        _CONNECT.append(("n", w, {kind: 6}))

_ent("cat",    "一只圆滚滚的橘猫",        "a plump round orange cat",
    ("猫", "猫咪", "小猫", "橘猫", "肥猫", "猫猫", "喵", "喵咪", "英短", "布偶", "猫崽"))
_ent("dog",    "一只摇着尾巴的小狗",      "a cheerful smiling puppy",
    ("狗", "狗狗", "小狗", "柴犬", "柯基", "柴", "金毛", "哈士奇", "汪", "狗崽"))
_ent("fish",   "一条跃出水面的鱼",        "a lively leaping fish",
    ("鱼", "金鱼", "小鱼", "热带鱼", "锦鲤", "鲤鱼", "鱼鱼"))
_ent("whale",  "一头温柔跃出水面的鲸",    "a gentle whale breaching the sea",
    ("鲸", "鲸鱼", "座头鲸", "蓝鲸", "虎鲸"))
_ent("dolphin","一只欢腾跃出海面的海豚",  "a playful dolphin leaping from the sea",
    ("海豚", "江豚", "海豚豚"))
_ent("penguin","一只摇摇摆摆的企鹅",      "a waddling round penguin",
    ("企鹅", "小企鹅", "帝企鹅", "胖企鹅", "南极"))
_ent("bird",   "一只振翅飞翔的小鸟",      "a tiny bird flying with spread wings",
    ("鸟", "小鸟", "麻雀", "燕子", "海鸥", "信天翁", "观鸟", "白头翁"))
_ent("fox",    "一只机灵俏皮的狐狸",      "a clever sly little fox",
    ("狐狸", "小狐狸", "火狐狸", "赤狐"))
_ent("rabbit", "一只竖着耳朵的白兔",      "a white floppy-eared bunny",
    ("兔子", "小兔子", "白兔", "垂耳兔", "兔兔"))
_ent("horse",  "一匹精神抖擞的马",        "a spirited galloping horse",
    ("马", "小马", "骏马", "小马驹", "黑马", "白马", "野马"))
_ent("bear",   "一只憨态可掬的熊",        "a cuddly chubby bear",
    ("熊", "小熊", "棕熊", "北极熊", "泰迪熊", "熊熊"))
_ent("hamster","一只圆滚滚的仓鼠",        "a tiny round hamster",
    ("仓鼠", "金丝熊", "鼠仔", "笼子里的仓鼠"))
_ent("rocket", "一枚拖着尾焰冲向夜空的火箭", "a cartoon rocket launching bright flame",
    ("火箭", "飞船", "航天器", "登月", "发射", "运载火箭"))
_ent("moon",   "一轮明镜似的圆月",        "a bright glowing full moon",
    ("月亮", "弯月", "圆月", "明月", "月色", "月牙", "月儿"))
_ent("sun",    "一轮暖洋洋的太阳",        "a warm radiant smiling sun",
    ("太阳", "日头", "朝阳", "烈日", "旭日", "日"))
_ent("star",   "一颗闪闪发光的小星星",    "a sparkling little star",
    ("星星", "星辰", "流星", "星", "星子"))
_ent("lighthouse","一座守在海边的灯塔",   "a lighthouse beaming over the sea",
    ("灯塔", "灯台", "指航灯"))
_ent("mountain","一带层层叠叠的山峦",     "layered misty mountain peaks",
    ("山", "山峰", "雪山", "群山", "峰峦", "青山", "群山", "山峦"))
_ent("house",  "一栋温馨的小房子",        "a cozy little cottage with a chimney",
    ("房子", "小屋", "屋舍", "蘑菇屋", "农舍", "小屋"))
_ent("castle", "一座童话里的城堡",        "a fairy-tale castle with turrets",
    ("城堡", "古堡", "童话城堡"))
_ent("tree",   "一棵生机勃勃的大树",      "a big leafy tree",
    ("树", "大树", "圣诞树", "樱花树", "松树", "小树苗"))
_ent("volcano","一座冒着烟的火山",        "a smoking volcano with glowing lava",
    ("火山", "活火山", "死火山"))
_ent("boat",   "一艘鼓起风帆的小船",      "a small sailboat sailing gentle waves",
    ("帆船", "小船", "船", "舰", "舟", "小舟", "木船", "渔船"))
_ent("submarine","一艘潜入深海的潜艇",    "a cute yellow submarine underwater",
    ("潜艇", "潜水艇"))
_ent("anchor", "一只停在海边的铁锚",      "an old ship anchor",
    ("锚", "船锚", "铁锚"))
_ent("sail",   "一面鼓满海风的风帆",      "a billowing white sail",
    ("风帆", "帆", "白帆"))
_ent("train",  "一列欢快的小火车",        "a cheerful little steam train",
    ("火车", "列车", "小火车", "蒸汽火车", "火车头"))
_ent("snowman","一个笑眯眯的雪人",        "a cheerful snowman with a scarf",
    ("雪人", "雪娃", "小雪人", "雪墩"))
_ent("rainbow","一道跨过天空的彩虹",      "a bright rainbow arcing across the sky",
    ("彩虹", "虹", "七彩桥"))
_ent("cloud",  "一朵软绵绵的白云",        "a fluffy white cloud",
    ("云", "云朵", "白云", "乌云", "云彩", "云朵朵"))
_ent("planet", "一颗五彩的行星",          "a colorful ringed planet",
    ("星球", "行星", "土星", "地球", "木星"))
_ent("ufo",    "一只神秘的天外来客飞碟",  "a mysterious glowing flying saucer",
    ("飞碟", "不明飞行物", "天外来客", "外星飞船"))
_ent("dragon", "一条盘旋的神龙",          "a majestic curling dragon",
    ("龙", "神龙", "巨龙", "飞龙", "东方龙"))
_ent("robot",  "一只圆头圆脑的小机器人",  "a cute round-headed robot",
    ("机器人", "机甲", "变形金刚", "机器仔"))
_ent("car",    "一辆复古可爱的小汽车",    "a cute vintage car",
    ("汽车", "小车", "跑车", "老爷车", "轿车"))
_ent("crab",   "一只横着爬的小螃蟹",      "a little red crab on the sand",
    ("螃蟹", "小螃蟹", "蟹", "帝王蟹"))
_ent("frog",   "一只蹲着打盹的小青蛙",    "a plump green frog on a leaf",
    ("青蛙", "小青蛙", "蛙", "牛蛙"))
_ent("cow",    "一头慢悠悠的奶牛",        "a gentle black-and-white cow",
    ("牛", "奶牛", "小牛", "黄牛", "哞牛"))
_ent("sheep",  "一只软绵绵的小羊",        "a fluffy white sheep",
    ("羊", "小羊", "绵羊", "山羊", "咩咩"))
_ent("dragonfly","一只点水的小蜻蜓",      "a delicate dragonfly over the water",
    ("蜻蜓", "小蜻蜓", "豆娘"))
_ent("pelican", "一只鹈鹕骑自行车",       "a pelican cycling",
    ("鹈鹕", "塘鹅", "pelican", "鹈"))   # v1.11: 删掉「滨鹬」—— 那是小型涉禽, 不是鹈鹕, 别让用户被画错

#  对外中文名 / 英文真图直取查表(主程序与旧代码都靠它们)
_OBJ_CN = {k: v["cn"] for k, v in _ENTITIES.items()}
_OBJ_EN = {k: v["en"] for k, v in _ENTITIES.items()}


#  ── 引擎注册表 ──────────────────────────────────────────────────────────────
#  每个主体可挂在某(或某几个)引擎名下; 引擎自带「词汇 → 主体」联想区间和专属渲染钩子。
#  引擎注册表按登记顺序查; 每主体只属于第一个挂载它的引擎(后登记不覆盖)。
_ENGINE_KIND = [
    ("pelican_ride", "鹈鹕骑行", ("pelican",)),
    ("animal_land",  "陆地动物", ("cat", "dog", "fox", "rabbit", "horse", "bear",
                                   "hamster", "cow", "sheep", "frog")),
    ("animal_sea",   "海洋动物", ("fish", "whale", "dolphin", "penguin", "crab",
                                   "dragonfly", "submarine")),
    ("animal_sky",   "天空动物", ("bird", "dragon")),
    ("space_sky",    "天空与太空", ("rocket", "moon", "sun", "star", "planet", "ufo",
                                     "cloud", "rainbow")),
    ("land_waters",  "地景与水上", ("lighthouse", "mountain", "volcano", "house",
                                     "castle", "tree", "snowman", "boat", "anchor",
                                     "sail", "train", "car", "robot")),
]
#  kind → 所属引擎名(用于回复里报"这是哪个引擎画的")
_KIND_ENGINE = {}
for _eid, _ename, _kinds in _ENGINE_KIND:
    for _k in _kinds:
        _KIND_ENGINE.setdefault(_k, _ename)


def register_engine(eid, ename, kinds, draw=None, parse=None):
    """插拔一个新引擎: 把一批主体挂到它名下, 可选提供专属几何 draw 与专属意图 parse。
       返回 True 表示挂载成功。未来独立引擎文件 import 后调用本函数即可自动注册。"""
    if not kinds:
        return False
    for _k in kinds:
        if _k not in _ENTITIES:
            continue
        _KIND_ENGINE.setdefault(_k, ename)
        if draw is not None:
            _DRAW[_k] = draw
        if parse is not None:
            _EXTRA_PARSE[_k] = parse
    return True


#  主体 → 专属意图解析拓展(个别主体可覆写通用气氛逻辑; 默认没有)
_EXTRA_PARSE = {}


def svgdream_scores(text):
    """联想打分: 名词/动词/形容词/场景全往主体上投票, 返回 {kind: 分数}。"""
    votes = {}
    t = (text or "").lower()
    for _typ, _w, _map in _CONNECT:
        if _w in t:
            for _k, _d in _map.items():
                if _k not in _ENTITIES:
                    continue
                votes[_k] = votes.get(_k, 0) + _d
    return votes


def svgdream_subject(text):
    """识别(或联想推断)主体。模糊意图也认: 只有动词/形容词/场景也能猜出最可能那个。"""
    votes = svgdream_scores(text)
    if not votes:
        return None
    return max(votes, key=votes.get)


def svgdream_is_request(text):
    """是不是「画/写/生成 一个XXX + 气氛」类诉求。算法/代码/知识/概念问答一律放行。
       注意: 这是「尽力画」不是「白名单查表」—— 主体哪怕从没见过(比如"有个人在编程"),
       也给通用骨架去拼, 而不是回一句"库里没有"。"""
    if is_pelican_bike(text):
        return True
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(k in t for k in ("算法", "代码", "函数", "程序", "脚本", "报错", "原理", "源码",
                            "实现一下", "是什么", "为什么", "怎么回答", "介绍", "讲解", "科普",
                            "生活习性", "吃什么", "栖息",
                            # v1.11 修路: "生成一篇作文" 这类是文本创作, 不是画图;
                            #   之前被 "生成" 两个字抢答成一张图, 作文路由永远轮不到。
                            "作文", "文章", "散文", "小说", "诗歌", "诗", "剧本", "文案",
                            "读后感", "演讲稿", "检讨书", "总结",
                            "报告", "周报", "月报", "简历", "简历", "邮件", "大纲", "提纲")):
        return False
    return any(k in t for k in ("画", "画个", "画一只", "画一幅", "画一座", "画一张", "画副", "画张",
                                "画一幅画", "帮我画", "给我画",
                                "来一张", "来张", "张图", "一张图",
                                "生成一个", "生成", "做一只", "做一栋", "做一座", "做一艘", "做一枚",
                                "弄个", "搞个", "要一只", "要一艘", "要一栋", "来一个", "来一张",
                                "脑补", "想象", "描绘"))


# ══════════════════════════════════════════════════════════════════════════════
# 通用意图: 时间/天气/情绪 + 主体, 凝成中文画面 + 英文真图 Prompt
# ══════════════════════════════════════════════════════════════════════════════
def _acc_phrase(t):
    if any(k in t for k in ("围巾", "围脖")):
        return "戴了条围巾", ", wearing a cozy scarf"
    if any(k in t for k in ("墨镜", "眼镜")):
        return "架着副墨镜", ", wearing cool sunglasses"
    if any(k in t for k in ("帽子", "凉帽")):
        return "戴顶小帽子", ", wearing a little hat"
    if any(k in t for k in ("花", "朵花", "戴花")):
        return "头顶别了朵花", ", with a small flower on top"
    return "", ""


#  ── 通用骨架分类: 库里没有的, 也能按「人/动物/机器/建筑」把身体现拼出来 ──
#  这是从「查找器」到「生成器」的关键: 认不出就归个类, 不再一句"库里没有"趴菜。
_ANIMAL_HINT = ("猫", "狗", "鸡", "鸭", "鹅", "牛", "马", "羊", "猪", "兔", "熊", "狐", "猴",
                "鱼", "鲸", "鸟", "鹰", "鹤", "龙", "龟", "鼠", "蛙", "鹿", "虎", "狮", "豹",
                "象", "蛇", "蜥", "虫", "蜂", "蝴蝶", "猴子", "老鼠", "仓鼠")
_PERSON_HINT = ("人", "小孩", "男孩", "女孩", "工人", "程序员", "医生", "老师", "作家", "厨师",
                "农民", "警察", "运动员", "爸爸", "妈妈", "少年", "少女", "学生", "骑士", "船长",
                "水手", "宇航员", "玩偶", "娃娃", "小人", "选手", "选手", "司机", "乘客")
_MACHINE_HINT = ("车", "船", "飞机", "火箭", "坦克", "火车", "潜艇", "自行车", "汽车", "飞船",
                 "挖掘机", "吊车", "车头", "机车", "摩托车", "卡车", "战车", "装甲车")
_STRUCT_HINT = ("房子", "屋", "塔", "桥", "城堡", "城", "楼", "庙", "亭", "墙", "山", "灯塔",
                "谷仓", "风车", "茅屋", "帐篷", "车站")


def svgdream_classify(text):
    """主体没见过时猜它是哪类, 好挑通用骨架拼。返回 human/animal/machine/structure/generic。"""
    t = (text or "").strip()
    if any(k in t for k in _PERSON_HINT):
        return "human"
    if any(k in t for k in _MACHINE_HINT):
        return "machine"
    if any(k in t for k in _STRUCT_HINT):
        return "structure"
    if any(k in t for k in _ANIMAL_HINT):
        return "animal"
    return "generic"


#  骨架类的默认画面描述(白名单主体用自己那套; 没见过的走这里)
_KIT_META = {
    "human":     ("一位专心的小人物", "a cute friendly cartoon person"),
    "animal":    ("一只可爱的小动物", "a cute cartoon animal"),
    "machine":   ("一辆可爱的小机器", "a cute cartoon machine"),
    "structure": ("一处可爱的小房子", "a cute little building"),
    "generic":   ("一幅可爱的小画",   "a cute cartoon illustration"),
}


def svgdream_intent(text, rnd=None):
    """读「画啥 + 什么气氛」: 时间/天气/情绪/挂件 + 主体, 出中文画面 + 英文真图 Prompt + 配色。"""
    rnd = rnd or random
    t = (text or "").strip()
    pal = ""
    if any(k in t for k in ("清晨", "黎明", "早上", "早晨")):
        pal = "清晨"
    elif any(k in t for k in ("正午", "中午", "午后", "大晴天")):
        pal = "正午"
    elif any(k in t for k in ("黄昏", "日落", "夕阳", "晚霞")):
        pal = "黄昏"
    elif any(k in t for k in ("傍晚",)):
        pal = "傍晚"
    elif any(k in t for k in ("深夜", "午夜", "夜晚", "晚上", "夜")):
        pal = "夜色"
    weather = ""
    if any(k in t for k in ("雪", "飘雪", "下雪", "雪地", "雪花")):
        weather = "snow"
    elif any(k in t for k in ("下雨", "雨天", "落雨", "雨")):
        weather = "rain"
    elif any(k in t for k in ("月亮", "月光", "月下", "月色", "星空", "星星")):
        weather = "moon"
    elif any(k in t for k in ("落花", "花瓣", "樱花", "繁花")):
        weather = "petals"
    mood = ""
    for _kw, _w in (("圣诞", "温馨的圣诞暖光"), ("雪", "清冽的冬日"), ("雨", "朦胧的雨幕"),
                    ("月亮", "静谧的月色"), ("星空", "深邃的星空"), ("夕阳", "暖洋洋的橙红"),
                    ("清晨", "清新的晨光"), ("樱花", "浪漫的落花"), ("花", "芬芳的花香"),
                    ("风", "动感的风"), ("黄昏", "慵懒的黄昏")):
        if _kw in t:
            mood = _w
            break
    palobj = None
    if weather in _WEATHER_PALS:
        palobj = _WEATHER_PALS[weather]
    elif pal in ("清晨", "正午", "黄昏", "夜色"):
        for _p in _PB_PALETTES:
            if _p["name"] == pal:
                palobj = _p
                break
    palobj = palobj or rnd.choice(_PB_PALETTES)
    kind = svgdream_subject(t)
    if kind is None:
        kind = svgdream_classify(t)          # 库里没有 → 归个类走通用骨架
    _cn, _en = _OBJ_CN.get(kind), _OBJ_EN.get(kind)
    if _cn is None:
        _cn, _en = _KIT_META.get(kind, ("一幅小画", "a cute illustration"))
    acc_cn, acc_en = _acc_phrase(t)
    theme = "骑行海边" if kind == "pelican" else (kind if kind in _OBJ_CN else _cn)
    light_en = {
        "snow": "falling snow, soft cold winter light", "rain": "light rainfall, cool blue-grey tones",
        "moon": "bright moonlight and stars", "petals": "cherry blossoms drifting, gentle light",
    }.get(weather, {
        "清晨": "soft golden morning glow", "正午": "bright midday sunshine",
        "黄昏": "warm golden sunset glow", "夜色": "calm night by the sea",
    }.get(pal, "gentle daylight"))
    style_en = rnd.choice((
        "flat vector illustration, clean pastel palette, soft shading",
        "charming storybook illustration, warm colors",
        "minimal cute style, rounded shapes, lively",
        "semi-realistic soft render, gentle gradients"))
    desc = "%s, %s%s" % (_cn, (mood + "的" if mood else "在"), acc_cn)
    if palobj["name"] not in ("清晨", "正午", "黄昏", "傍晚", "夜色"):
        _scene = "雪天" if weather == "snow" else "雨天" if weather == "rain" else "月夜" if weather == "moon" else "花海" if weather == "petals" else "海边"
    else:
        _scene = palobj["name"]
    prompt = ("%s%s%s, %s, %s, %s, high quality, 4k, crisp edges"
              % (_en, acc_en, "", light_en, style_en, palobj["name"]))
    return {"subject": kind, "weather": weather, "theme": _scene, "mood": mood,
            "desc_cn": desc.strip("，,"), "prompt_en": prompt, "pal_override": palobj,
            "palette_name": palobj["name"], "seed": rnd.random(), "raw": t}


# ══════════════════════════════════════════════════════════════════════════════
# 通用画布 + CSS 动画 + 主体几何
# ══════════════════════════════════════════════════════════════════════════════
_SVG_STYLE = ("<style>"
              "@keyframes sway{0%,100%{transform:rotate(0)}50%{transform:rotate(6deg)}}"
              "@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}"
              "@keyframes swim{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-7px) rotate(-3deg)}}"
              "@keyframes flick{0%,100%{opacity:1}50%{opacity:.5}}"
              "@keyframes spin360{from{transform:rotate(0)}to{transform:rotate(360deg)}}"
              "</style>")


#  ── 可插拔几何分发表 ──────────────────────────────────────────────────────────
#  每个主体一个绘制函数(draw(rnd, pal)) → 返回一段几何 SVG 字符串。
#  加新主体: 往 _DRAW 里丢一个函数即可; 独立引擎文件可直接 _DRAW[k][=fn] 或 register_engine()。
_DRAW = {}


def _draw_subject(kind, rnd, pal):
    """按主体画一层几何 + CSS 动画(坐标落在 720x440 画布)。可插拔, 认不出就走兜底球。"""
    return _DRAW.get(kind, _DRAW["__default__"])(rnd, pal)


#  —— 陆地动物 ——
_DRAW["cat"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 4s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="352" cy="360" rx="52" ry="34" fill="#f2b04c" stroke="#d99b3f" stroke-width="2.5"/>'
    '<circle cx="330" cy="302" r="26" fill="#f2b04c" stroke="#d99b3f" stroke-width="2.5"/>'
    '<path d="M312,282 L318,258 L334,274 z" fill="#f2b04c"/><path d="M344,278 L352,254 L362,272 z" fill="#f2b04c"/>'
    '<circle cx="322" cy="298" r="3.4" fill="#26303f"/><circle cx="340" cy="298" r="3.4" fill="#26303f"/>'
    '<path d="M329,312 L331,306 L333,312 z" fill="#e07a7a"/><path d="M336,296 q4,7 7,2" stroke="#26303f" fill="none" stroke-width="1.6"/>'
    '<path d="M300,300 q-14,4 -20,12 q16,6 20,-6 z" fill="#f2b04c"/><path d="M344,298 q-4,-16 -16,-18" stroke="#d99b3f" fill="none" stroke-width="3" stroke-linecap="round"/></g>')
_DRAW["dog"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.4s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="354" cy="364" rx="54" ry="32" fill="#c8956a" stroke="#a97a52" stroke-width="2.5"/>'
    '<circle cx="322" cy="306" r="25" fill="#c8956a" stroke="#a97a52" stroke-width="2.5"/>'
    '<path d="M300,286 q-16,2 -28,16 q22,4 30,-8 z" fill="#c8956a" stroke="#a97a52"/><path d="M330,290 q-14,-18 -6,-34 q10,-6 18,0 q2,14 -8,26 z" fill="#c8956a" stroke="#a97a52"/>'
    '<ellipse cx="366" cy="390" rx="9" ry="5" fill="#a97a52" transform="rotate(-14 366 390)"/>'
    '<circle cx="315" cy="302" r="3.2" fill="#2a2a2a"/><circle cx="331" cy="302" r="3.2" fill="#2a2a2a"/>'
    '<ellipse cx="323" cy="312" rx="6" ry="4" fill="#e07a7a"/></g>')
_DRAW["fox"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="356" cy="362" rx="46" ry="32" fill="#f2903f" stroke="#d97b2e" stroke-width="2.5"/>'
    '<circle cx="322" cy="306" r="24" fill="#f2903f" stroke="#d97b2e" stroke-width="2.5"/>'
    '<path d="M305,296 l-10,-28 l14,12 l14,-20 l12,26 z" fill="#f2903f" stroke="#d97b2e"/>'
    '<path d="M312,288 l-4,-16 l8,6 l8,-14" fill="#f2d4a8" stroke="#d97b2e" transform="rotate(-6 320 300)"/>'
    '<circle cx="315" cy="302" r="3" fill="#26303f"/><circle cx="331" cy="302" r="3" fill="#26303f"/>'
    '<path d="M352,370 q8,18 4,32 q-8,-6 -10,-18 z" fill="#fdf6e8"/><path d="M356,404 q0,8 -12,8 q6,-10 12,-8 z" fill="#f2903f"/></g>')
_DRAW["rabbit"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="356" cy="368" rx="44" ry="36" fill="#f7f2ea" stroke="#d8cfc2" stroke-width="2.5"/>'
    '<circle cx="330" cy="308" r="24" fill="#f7f2ea" stroke="#d8cfc2" stroke-width="2.5"/>'
    '<path d="M318,290 q-6,-34 2,-48 q8,14 6,48 z" fill="#f7f2ea" stroke="#d8cfc2"/><path d="M344,288 q-2,-36 8,-44 q6,18 2,46 z" fill="#f7f2ea" stroke="#d8cfc2"/>'
    '<ellipse cx="318" cy="270" rx="5" ry="14" fill="#f7c7d0"/><ellipse cx="340" cy="266" rx="5" ry="14" fill="#f7c7d0"/>'
    '<circle cx="323" cy="304" r="3" fill="#26303f"/><circle cx="339" cy="304" r="3" fill="#26303f"/>'
    '<path d="M329,316 q3,5 6,0" stroke="#26303f" fill="none" stroke-width="1.5"/></g>')
_DRAW["horse"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.2s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="360" cy="330" rx="74" ry="26" fill="#9a6b46" stroke="#7d5636" stroke-width="2.5"/>'
    '<circle cx="282" cy="280" r="26" fill="#9a6b46" stroke="#7d5636" stroke-width="2.5"/>'
    '<path d="M300,266 l14,-24 l-2,22" fill="#9a6b46" stroke="#7d5636"/><path d="M286,252 l8,-20 l10,16" fill="#5b3d28"/>'
    '<rect x="264" y="288" width="40" height="8" rx="4" fill="#5b3d28"/>'
    '<circle cx="276" cy="274" r="3" fill="#1d2733"/><circle cx="292" cy="274" r="3" fill="#1d2733"/>'
    '<rect x="300" y="348" width="14" height="36" rx="6" fill="#7d5636"/><rect x="330" y="348" width="14" height="36" rx="6" fill="#7d5636"/>'
    '<rect x="390" y="348" width="14" height="36" rx="6" fill="#7d5636"/><rect x="418" y="348" width="14" height="36" rx="6" fill="#7d5636"/></g>')
_DRAW["bear"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="358" cy="366" rx="58" ry="38" fill="#b07a4a" stroke="#8f623c" stroke-width="2.5"/>'
    '<circle cx="348" cy="300" r="28" fill="#b07a4a" stroke="#8f623c" stroke-width="2.5"/>'
    '<circle cx="318" cy="276" r="12" fill="#b07a4a"/><circle cx="376" cy="276" r="12" fill="#b07a4a"/>'
    '<circle cx="318" cy="276" r="5" fill="#8f623c"/><circle cx="376" cy="276" r="5" fill="#8f623c"/>'
    '<ellipse cx="330" cy="320" rx="12" ry="9" fill="#f2c89a"/><ellipse cx="368" cy="320" rx="12" ry="9" fill="#f2c89a"/>'
    '<circle cx="338" cy="296" r="3.4" fill="#1d2733"/><circle cx="358" cy="296" r="3.4" fill="#1d2733"/>'
    '<path d="M342,310 q6,6 12,0" stroke="#26303f" fill="none" stroke-width="2"/></g>')
_DRAW["hamster"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 2.6s ease-in-out infinite;transform-origin:360px 380px">'
    '<ellipse cx="360" cy="352" rx="40" ry="34" fill="#e8b572" stroke="#cf9c5c" stroke-width="2.5"/>'
    '<circle cx="360" cy="300" r="24" fill="#f0c98a" stroke="#cf9c5c" stroke-width="2.5"/>'
    '<circle cx="342" cy="288" r="7" fill="#f0c98a"/><circle cx="380" cy="288" r="7" fill="#f0c98a"/>'
    '<ellipse cx="330" cy="316" rx="13" ry="11" fill="#fdf2df" stroke="#e8b572"/><ellipse cx="390" cy="316" rx="13" ry="11" fill="#fdf2df" stroke="#e8b572"/>'
    '<circle cx="350" cy="298" r="3" fill="#1d2733"/><circle cx="370" cy="298" r="3" fill="#1d2733"/>'
    '<path d="M358,310 q2,4 4,0" stroke="#26303f" fill="none" stroke-width="1.6"/></g>')
_DRAW["cow"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.6s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="356" cy="366" rx="60" ry="34" fill="#f7f7f7" stroke="#c9cdd4" stroke-width="2.5"/>'
    '<circle cx="336" cy="330" r="26" fill="#f7f7f7" stroke="#c9cdd4" stroke-width="2.5"/>'
    '<path d="M322,306 l-8,-26 l4,-2 l12,20 z" fill="#f7f7f7" stroke="#c9cdd4"/><path d="M350,302 l8,-28 l-2,-2 l-14,22 z" fill="#f7f7f7" stroke="#c9cdd4"/>'
    '<circle cx="397" cy="348" r="10" fill="#e4574f"/><path d="M320,356 q16,10 34,0" fill="#e4574f"/>'
    '<circle cx="396" cy="348" r="4" fill="#f2c89a"/><circle cx="388" cy="348" r="3" fill="#f2c89a"/>'
    '<ellipse cx="312" cy="340" rx="6" ry="4" fill="#f2c89a"/><ellipse cx="358" cy="340" rx="6" ry="4" fill="#f2c89a"/>'
    '<circle cx="404" cy="354" r="3" fill="#26303f"/><circle cx="410" cy="348" r="3" fill="#26303f"/></g>')
_DRAW["sheep"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.6s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="360" cy="366" rx="58" ry="36" fill="#fdf9f0" stroke="#e3dccd" stroke-width="2.5"/>'
    '<circle cx="330" cy="342" r="13" fill="#cfd6de" transform="translate(-26,-4)"/><circle cx="390" cy="342" r="13" fill="#cfd6de" transform="translate(0,-14)"/><circle cx="360" cy="340" r="13" fill="#cfd6de" transform="translate(0,8)"/>'
    '<circle cx="348" cy="312" r="20" fill="#4a4a4a" stroke="#333" stroke-width="2.5"/>'
    '<circle cx="356" cy="302" r="5" fill="#1d2733"/><circle cx="372" cy="302" r="5" fill="#1d2733"/>'
    '<path d="M360,310 l3,5 l-6,0 z" fill="#f2a63c"/><rect x="326" y="344" width="8" height="20" rx="4" fill="#333"/><rect x="378" y="344" width="8" height="20" rx="4" fill="#333"/></g>')
_DRAW["frog"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="360" cy="370" rx="52" ry="34" fill="#7cb85c" stroke="#5c9a44" stroke-width="2.5"/>'
    '<circle cx="332" cy="318" r="20" fill="#7cb85c"/><circle cx="390" cy="318" r="20" fill="#7cb85c"/>'
    '<circle cx="328" cy="314" r="7" fill="#ffffff"/><circle cx="328" cy="314" r="4" fill="#1d2733"/>'
    '<circle cx="386" cy="314" r="7" fill="#ffffff"/><circle cx="386" cy="314" r="4" fill="#1d2733"/>'
    '<path d="M354,336 q8,10 16,0 l-3,8 q-5,6 -10,0 z" fill="#5c8a44"/>'
    '<ellipse cx="306" cy="398" rx="14" ry="8" fill="#5c9a44"/><ellipse cx="416" cy="398" rx="14" ry="8" fill="#5c9a44"/></g>')
#  —— 海洋动物 ——
_DRAW["fish"] = lambda rnd, pal: (
    '<g class="swim" style="animation:swim 3s ease-in-out infinite;transform-origin:360px 330px">'
    '<path d="M300,320 q40,-28 74,0 q-34,28 -74,0 z" fill="#f25f7a" stroke="#d94a66" stroke-width="2.5"/>'
    '<path d="M374,320 l22,-16 l-4,16 z" fill="#f25f7a"/><circle cx="316" cy="312" r="3.4" fill="#26303f"/>'
    '<path d="M316,330 q6,6 12,6" stroke="#fff2f4" fill="none" stroke-width="2"/></g>')
_DRAW["whale"] = lambda rnd, pal: (
    '<g class="swim" style="animation:swim 3.6s ease-in-out infinite;transform-origin:360px 330px">'
    '<ellipse cx="340" cy="330" rx="92" ry="50" fill="#5b8fd6" stroke="#3f6fb6" stroke-width="3"/>'
    '<path d="M424,318 q26,8 34,16 q-6,12 -34,12 z" fill="#5b8fd6"/>'
    '<circle cx="296" cy="316" r="5" fill="#1d2733"/><path d="M298,344 q12,8 24,10" stroke="#9cc2ec" fill="none" stroke-width="3"/>'
    '<path d="M368,296 q-2,-14 6,-20 q8,4 8,14 z" fill="#9cc2ec"/></g>')
_DRAW["dolphin"] = lambda rnd, pal: (
    '<g class="swim" style="animation:swim 3.2s ease-in-out infinite;transform-origin:360px 320px">'
    '<path d="M290,330 q70,-52 130,-8 q-10,-14 -14,-40 q-4,26 -22,46 q-26,14 -58,4 q-10,-6 -16,2 q12,18 40,20 q22,2 60,-10 q18,-6 30,-24 l16,8 q-12,22 -34,32 q-42,18 -78,8 q-34,-8 -58,-20 q-20,4 -36,-6 z" fill="#8fb8e8" stroke="#6a96c9" stroke-width="3"/>'
    '<circle cx="352" cy="286" r="5" fill="#1d2733"/><path d="M344,298 q8,8 16,4" stroke="#eaf3fc" fill="none" stroke-width="3"/></g>')
_DRAW["penguin"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.2s ease-in-out infinite;transform-origin:360px 400px">'
    '<ellipse cx="360" cy="368" rx="34" ry="48" fill="#2c3850" stroke="#202a3d" stroke-width="2.5"/>'
    '<ellipse cx="360" cy="380" rx="22" ry="32" fill="#f7f7f7"/>'
    '<circle cx="360" cy="300" r="22" fill="#2c3850"/><ellipse cx="360" cy="300" rx="13" ry="14" fill="#f7f7f7"/>'
    '<circle cx="353" cy="304" r="3" fill="#111"/><circle cx="367" cy="304" r="3" fill="#111"/>'
    '<path d="M360,312 l3,5 l-6,0 z" fill="#f2a63c"/></g>')
_DRAW["crab"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.4s ease-in-out infinite;transform-origin:360px 380px">'
    '<ellipse cx="360" cy="354" rx="36" ry="26" fill="#e4574f" stroke="#c23f38" stroke-width="2.5"/>'
    '<path d="M300,340 q-26,4 -34,18 q24,2 36,-6 z" fill="#e4574f"/><path d="M420,340 q26,4 34,18 q-24,2 -36,-6 z" fill="#e4574f"/>'
    '<circle cx="344" cy="344" r="4" fill="#fff"/><circle cx="344" cy="344" r="2" fill="#1d2733"/>'
    '<circle cx="378" cy="344" r="4" fill="#fff"/><circle cx="378" cy="344" r="2" fill="#1d2733"/>'
    '<path d="M358,356 l-2,10 l-10,-4 z" fill="#c23f38"/><path d="M362,356 l2,10 l10,-4 z" fill="#c23f38"/></g>')
_DRAW["dragonfly"] = lambda rnd, pal: (
    '<g class="bob" style="animation:bob 2s ease-in-out infinite;transform-origin:360px 220px">'
    '<ellipse cx="360" cy="220" rx="8" ry="30" fill="#6aa0d0"/>'
    '<circle cx="352" cy="196" r="7" fill="#2c3850"/><circle cx="362" cy="196" r="7" fill="#2c3850"/>'
    '<path d="M368,214 q26,-20 48,-26 q-6,24 -34,34 z" fill="#bfe0ff" opacity=".85"/>'
    '<path d="M352,214 q-26,-18 -46,-24 q6,24 32,34 z" fill="#bfe0ff" opacity=".85"/>'
    '<path d="M358,244 q14,2 26,-4 q-12,4 -26,4 z" fill="#4a6f9e"/></g>')
_DRAW["submarine"] = lambda rnd, pal: (
    '<g class="swim" style="animation:swim 4s ease-in-out infinite;transform-origin:360px 330px">'
    '<ellipse cx="360" cy="330" rx="86" ry="38" fill="#ffd95e" stroke="#e0b93f" stroke-width="3"/>'
    '<path d="M330,304 l60,-16 l16,26 z" fill="#f5a623"/><circle cx="352" cy="318" r="18" fill="#8fd0ff" stroke="#4a90c9" stroke-width="3"/>'
    '<circle cx="360" cy="318" r="7" fill="#1d2733"/><rect x="300" y="352" width="46" height="10" rx="5" fill="#4a90c9"/>'
    '<rect x="380" y="352" width="34" height="8" rx="4" fill="#4a90c9"/>'
    '<path d="M430,344 q8,-10 2,-24" stroke="#e0b93f" fill="none" stroke-width="3"/></g>')
#  —— 天空动物 ——
_DRAW["bird"] = lambda rnd, pal: (
    '<g class="bob" style="animation:bob 1.8s ease-in-out infinite;transform-origin:360px 200px">'
    '<ellipse cx="356" cy="208" rx="34" ry="24" fill="#ef9930" stroke="#d6801f" stroke-width="2.5"/>'
    '<circle cx="326" cy="192" r="16" fill="#ef9930" stroke="#d6801f" stroke-width="2.5"/>'
    '<path d="M312,190 l-10,-4 l8,8 z" fill="#f2a63c"/>'
    '<circle cx="326" cy="190" r="2.6" fill="#1d2733"/>'
    '<path d="M352,206 q20,-14 44,-18 q4,8 -10,16 q-14,4 -34,2 z" fill="#d6801f" style="animation:flick .8s ease-in-out infinite"/>'
    '<path d="M358,222 q20,10 40,8 q-4,-8 -16,-12 q-12,-2 -24,4 z" fill="#d6801f"/></g>')
_DRAW["dragon"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3s ease-in-out infinite;transform-origin:360px 240px">'
    '<path d="M300,240 q40,-36 84,-24 q26,8 48,-14 q-6,22 -6,36 q0,20 10,30 q-24,-2 -44,2 q-18,10 -30,26 q-30,6 -54,-8 q-14,-22 -8,-48 z" fill="#43a047" stroke="#2e7d32" stroke-width="3"/>'
    '<path d="M250,252 q-26,-16 -44,-8 q14,-8 18,-22 q12,6 26,12 z" fill="#2e7d32"/>'
    '<circle cx="344" cy="216" r="5" fill="#fffbe6"/><circle cx="344" cy="216" r="2.6" fill="#1d2733"/><circle cx="360" cy="216" r="5" fill="#fffbe6"/><circle cx="360" cy="216" r="2.6" fill="#1d2733"/>'
    '<path d="M348,208 l-6,-18 l12,4 z" fill="#e4574f" stroke="#c23f38"/>'
    '<circle cx="306" cy="264" r="16" fill="#43a047" stroke="#2e7d32"/><path d="M306,280 l10,22 l12,-20" fill="#2e7d32"/></g>')
#  —— 天空与太空 ——
_DRAW["rocket"] = lambda rnd, pal: (
    '<g class="bob" style="animation:bob 1.2s ease-in-out infinite;transform-origin:360px 250px">'
    '<path d="M348,120 L372,120 L368,150 L352,150 z" fill="#e4574f"/>'
    '<rect x="348" y="120" width="24" height="150" rx="8" fill="#cfd8e3" stroke="#b6c2cc" stroke-width="2.5"/>'
    '<circle cx="360" cy="176" r="13" fill="#243560" stroke="#b6c2cc" stroke-width="3"/>'
    '<path d="M348,268 l-22,26 l28,0 l-6,-26 z" fill="#e4574f"/><path d="M372,268 l22,26 l-28,0 l6,-26 z" fill="#e4574f"/>'
    '<path d="M350,286 q10,16 20,0 z" fill="#ffb454" style="animation:flick .5s ease-in-out infinite"/></g>')
_DRAW["moon"] = lambda rnd, pal: (
    '<g><circle cx="360" cy="185" r="96" fill="#f6eec4" stroke="#e6d9a6" stroke-width="3"/>'
    '<circle cx="330" cy="160" r="18" fill="#eee2b0" opacity=".7"/><circle cx="392" cy="208" r="26" fill="#eee2b0" opacity=".7"/>'
    '<circle cx="360" cy="240" r="12" fill="#eee2b0" opacity=".7"/>'
    '<circle cx="360" cy="185" r="108" fill="#fff5d6" opacity=".25" style="animation:flick 4s ease-in-out infinite"/></g>')
_DRAW["sun"] = lambda rnd, pal: (
    '<g style="animation:spin360 30s linear infinite;transform-origin:360px 200px">'
    '<circle cx="360" cy="200" r="54" fill="%s" stroke="#e8a33c" stroke-width="3"/>'
    '<path d="M360,120 l0,-30 M360,280 l0,30 M280,200 l-30,0 M440,200 l30,0 M302,142 l-21,-21 M418,258 l21,21 M418,142 l21,-21 M302,258 l-21,21" stroke="%s" stroke-width="7" stroke-linecap="round"/>'
    '<circle cx="340" cy="188" r="6" fill="#d97b2e"/><circle cx="380" cy="188" r="6" fill="#d97b2e"/>'
    '<path d="M342,212 q18,14 36,0" stroke="#d97b2e" fill="none" stroke-width="5" stroke-linecap="round"/></g>' % (pal["core"], pal["sun"]))
_DRAW["star"] = lambda rnd, pal: (
    '<g style="animation:flick 2s ease-in-out infinite">'
    '<path d="M360,120 l18,40 l44,2 l-34,28 l12,44 l-40,-26 l-40,26 l12,-44 l-34,-28 l44,-2 z" fill="%s" stroke="%s" stroke-width="3" stroke-linejoin="round"/>'
    '<circle cx="360" cy="190" r="72" fill="%s" opacity=".25"/></g>' % (pal["core"], pal["sun"], pal["core"]))
_DRAW["planet"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 6s ease-in-out infinite;transform-origin:360px 200px">'
    '<circle cx="360" cy="200" r="76" fill="#e0b45e" stroke="#c4943f" stroke-width="3"/>'
    '<path d="M286,168 q74,-24 148,0" stroke="#c4943f" fill="none" stroke-width="4"/><path d="M286,234 q74,26 148,0" stroke="#c4943f" fill="none" stroke-width="4"/>'
    '<ellipse cx="360" cy="200" rx="118" ry="18" fill="none" stroke="#7a86a8" stroke-width="5" transform="rotate(-14 360 200)"/>'
    '<circle cx="330" cy="184" r="8" fill="#c4943f"/><circle cx="392" cy="212" r="10" fill="#8fb0cc"/></g>')
_DRAW["ufo"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 2.6s ease-in-out infinite;transform-origin:360px 220px">'
    '<ellipse cx="360" cy="224" rx="86" ry="26" fill="#b8ced8" stroke="#8aa7b5" stroke-width="3"/>'
    '<path d="M320,224 q40,-60 80,0 z" fill="#a9c6d6" stroke="#8aa7b5" stroke-width="3"/>'
    '<circle cx="332" cy="238" r="5" fill="#7adbe8"/><circle cx="360" cy="242" r="5" fill="#7adbe8"/><circle cx="388" cy="238" r="5" fill="#7adbe8"/>'
    '<path d="M286,224 q-16,10 -4,16 q12,4 20,-6 z" fill="#8aa7b5"/><path d="M434,224 q16,10 4,16 q-12,4 -20,-6 z" fill="#8aa7b5"/>'
    '<ellipse cx="360" cy="252" rx="30" ry="8" fill="#d98be0" style="animation:flick 1.4s ease-in-out infinite"/></g>')
_DRAW["cloud"] = lambda rnd, pal: (
    '<g style="animation:flick 5s ease-in-out infinite">'
    '<ellipse cx="360" cy="200" rx="90" ry="44" fill="#ffffff" stroke="#dfe6ee" stroke-width="3"/>'
    '<circle cx="300" cy="196" r="34" fill="#ffffff"/><circle cx="372" cy="182" r="42" fill="#ffffff"/><circle cx="416" cy="206" r="28" fill="#ffffff"/>'
    '<path d="M300,214 q60,14 120,-2" stroke="#dfe6ee" fill="none" stroke-width="2" opacity=".6"/></g>')
_DRAW["rainbow"] = lambda rnd, pal: (
    '<g>'
    '<path d="M200,310 q160,-170 320,0" fill="none" stroke="#e4574f" stroke-width="16"/>'
    '<path d="M212,310 q148,-156 296,0" fill="none" stroke="#f2a63c" stroke-width="16"/>'
    '<path d="M224,310 q136,-142 272,0" fill="none" stroke="#ffd95e" stroke-width="16"/>'
    '<path d="M236,310 q124,-128 248,0" fill="none" stroke="#7cb85c" stroke-width="16"/>'
    '<path d="M248,310 q112,-114 224,0" fill="none" stroke="#5b8fd6" stroke-width="16"/>'
    '<path d="M260,310 q100,-100 200,0" fill="none" stroke="#a06ab8" stroke-width="16"/></g>')
#  —— 地景与水上 ——
_DRAW["lighthouse"] = lambda rnd, pal: (
    '<g>'
    '<path d="M322,404 L372,404 L376,250 L318,250 z" fill="#f3f6fb" stroke="#cfd8e3" stroke-width="3"/>'
    '<path d="M318,250 L318,290 L376,290 L376,250 z" fill="#e4574f"/>'
    '<circle cx="347" cy="216" r="16" fill="#2a3550" stroke="#1c2438" stroke-width="3"/>'
    '<circle cx="347" cy="216" r="9" fill="#ffe27a" style="animation:flick 1.1s ease-in-out infinite"/>'
    '<path d="M320,60 L374,60 L347,120 z" fill="#e4574f"/>'
    '<rect x="342" y="320" width="10" height="22" rx="5" fill="#2a3550"/>'
    '<circle cx="347" cy="360" r="4" fill="#cfd8e3"/></g>')
_DRAW["mountain"] = lambda rnd, pal: (
    '<g fill="#7fa3c2" stroke="#5c82a6" stroke-width="2">'
    '<path d="M70,350 L210,150 L330,350 z"/><path d="M250,350 L420,90 L560,350 z" fill="#8fb0cc"/>'
    '<path d="M286,220 L352,330 L268,330 z" fill="#5c82a6"/>'
    '<path d="M360,180 L420,90 L404,210 L376,210 z" fill="#eef3f8"/></g>')
_DRAW["volcano"] = lambda rnd, pal: (
    '<g>'
    '<path d="M250,360 L360,130 L472,360 z" fill="#6a4a38" stroke="#52392b" stroke-width="3"/>'
    '<path d="M378,360 L360,130 L320,360 z" fill="#7d5a45" opacity=".6"/>'
    '<ellipse cx="360" cy="140" rx="34" ry="14" fill="#4a3529"/>'
    '<ellipse cx="360" cy="142" rx="20" ry="8" fill="#ff9d45" style="animation:flick .6s ease-in-out infinite"/>'
    '<path d="M352,110 q-6,-22 8,-36 q-6,16 -2,30 M368,104 q2,-18 14,-26 q-2,14 -8,22" stroke="#8a8f98" fill="none" stroke-width="6" stroke-linecap="round" opacity=".8"/></g>')
_DRAW["house"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 5s ease-in-out infinite;transform-origin:360px 360px">'
    '<path d="M250,340 L360,220 L472,340 z" fill="#e4574f" stroke="#c23f38" stroke-width="3"/>'
    '<rect x="262" y="340" width="196" height="70" fill="#f3e6c9" stroke="#d8c9a6" stroke-width="3"/>'
    '<rect x="300" y="360" width="44" height="50" fill="#8a5a3a" rx="3"/>'
    '<rect x="344" y="330" width="22" height="34" rx="8" fill="#8fd0ff" stroke="#4a90c9" stroke-width="2"/>'
    '<rect x="388" y="330" width="22" height="34" rx="8" fill="#8fd0ff" stroke="#4a90c9" stroke-width="2"/>'
    '<path d="M430,300 q6,-30 18,-38 q2,6 -4,18 q-2,22 -14,20 z" fill="#e8e8e8" stroke="#c9c9c9" stroke-width="2"/></g>')
_DRAW["castle"] = lambda rnd, pal: (
    '<g>'
    '<rect x="250" y="280" width="220" height="90" fill="#d8cfc2" stroke="#b8ad9c" stroke-width="3"/>'
    '<rect x="210" y="250" width="54" height="110" fill="#cfc4b4" stroke="#b8ad9c" stroke-width="3"/>'
    '<rect x="456" y="250" width="54" height="110" fill="#cfc4b4" stroke="#b8ad9c" stroke-width="3"/>'
    '<path d="M200,250 l42,-44 l42,44 z" fill="#e2605a" stroke="#b8ad9c" stroke-width="3"/><path d="M446,250 l42,-44 l42,44 z" fill="#e2605a" stroke="#b8ad9c" stroke-width="3"/>'
    '<circle cx="360" cy="250" r="18" fill="#7aa2d6" stroke="#b8ad9c" stroke-width="3"/>'
    '<path d="M330,250 l30,-40 l30,40 z" fill="#e2605a"/><rect x="390" y="334" width="34" height="36" rx="10" fill="#7d5a45"/>'
    '<rect x="340" y="334" width="40" height="36" rx="12" fill="#7d5a45"/></g>')
_DRAW["tree"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 5s ease-in-out infinite;transform-origin:360px 380px">'
    '<rect x="346" y="320" width="28" height="70" rx="10" fill="#7d5a45" stroke="#5c3a24" stroke-width="2"/>'
    '<circle cx="360" cy="270" r="58" fill="#57a85c" stroke="#3f8c46" stroke-width="3"/>'
    '<circle cx="320" cy="288" r="34" fill="#63b669"/><circle cx="400" cy="288" r="34" fill="#63b669"/>'
    '<circle cx="352" cy="282" r="6" fill="#e4574f"/><circle cx="380" cy="296" r="5" fill="#f2a63c"/></g>')
_DRAW["snowman"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.6s ease-in-out infinite;transform-origin:360px 400px">'
    '<circle cx="360" cy="368" r="40" fill="#ffffff" stroke="#cfd8e3" stroke-width="2.5"/>'
    '<circle cx="360" cy="300" r="28" fill="#fdfdfd" stroke="#cfd8e3" stroke-width="2.5"/>'
    '<path d="M342,268 L344,244 L366,248 L362,270 z" fill="#2a3550"/>'
    '<path d="M360,300 l-10,-4 l4,12 z" fill="#f2a63c"/>'
    '<circle cx="350" cy="292" r="3" fill="#26303f"/><circle cx="370" cy="292" r="3" fill="#26303f"/>'
    '<rect x="332" y="352" width="56" height="8" rx="4" fill="#e4574f"/>'
    '<circle cx="346" cy="340" r="3" fill="#2a3550"/><circle cx="360" cy="342" r="3" fill="#2a3550"/><circle cx="374" cy="344" r="3" fill="#2a3550"/></g>')
_DRAW["boat"] = lambda rnd, pal: (
    '<g class="swim" style="animation:swim 4s ease-in-out infinite;transform-origin:360px 350px">'
    '<path d="M300,368 L420,368 L396,404 L324,404 z" fill="#b97f4f" stroke="#96643c" stroke-width="3"/>'
    '<line x1="360" y1="368" x2="360" y2="300" stroke="#5c3a24" stroke-width="4"/>'
    '<path d="M360,300 L406,342 L360,342 z" fill="#f3f6fb" stroke="#cfd8e3" stroke-width="2"/>'
    '<circle cx="360" cy="332" r="3" fill="#e4574f"/></g>')
_DRAW["anchor"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 4s ease-in-out infinite;transform-origin:360px 300px">'
    '<line x1="360" y1="120" x2="360" y2="300" stroke="#8a5a3a" stroke-width="7" stroke-linecap="round"/>'
    '<circle cx="360" cy="120" r="12" fill="none" stroke="#8a5a3a" stroke-width="7"/>'
    '<path d="M300,200 q60,-40 120,0 q-20,60 -52,84 q-6,-20 6,-40 q-10,18 -26,22 q-6,-26 8,-46 q-6,16 -22,22 q16,-30 6,-52 z" fill="#a06a4a" stroke="#7d5636" stroke-width="4"/></g>')
_DRAW["sail"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 4s ease-in-out infinite;transform-origin:360px 300px">'
    '<line x1="290" y1="360" x2="360" y2="120" stroke="#5c3a24" stroke-width="6"/>'
    '<path d="M364,130 L420,330 L290,330 q74,-120 74,-200 z" fill="#f7fbfe" stroke="#cfd8e3" stroke-width="3"/>'
    '<path d="M364,130 L290,330 L364,330 z" fill="#e8f2fb" opacity=".7"/></g>')
_DRAW["train"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 4s ease-in-out infinite;transform-origin:360px 350px">'
    '<rect x="240" y="286" width="120" height="52" rx="8" fill="#e4574f" stroke="#c23f38" stroke-width="3"/>'
    '<rect x="360" y="306" width="54" height="32" rx="4" fill="#f2a63c" stroke="#d6801f" stroke-width="3"/>'
    '<rect x="256" y="298" width="20" height="26" rx="5" fill="#8fd0ff" stroke="#4a90c9" stroke-width="2"/><rect x="284" y="298" width="20" height="26" rx="5" fill="#8fd0ff" stroke="#4a90c9" stroke-width="2"/><rect x="312" y="298" width="20" height="26" rx="5" fill="#8fd0ff" stroke="#4a90c9" stroke-width="2"/>'
    '<circle cx="282" cy="348" r="12" fill="#3a4046" stroke="#26303f" stroke-width="3"/><circle cx="372" cy="348" r="12" fill="#3a4046" stroke="#26303f" stroke-width="3"/>'
    '<path d="M240,286 q-8,-26 10,-34 q-2,12 8,20 q10,6 18,6" fill="#e8e8e8" stroke="#c9c9c9" stroke-width="2"/></g>')
_DRAW["car"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 4s ease-in-out infinite;transform-origin:360px 350px">'
    '<path d="M248,352 L252,310 L330,310 L358,286 L436,286 L462,352 z" fill="#5799d6" stroke="#3f7bb2" stroke-width="3"/>'
    '<path d="M330,310 L358,292 L348,310 z" fill="#3f7bb2"/>'
    '<rect x="300" y="300" width="44" height="14" rx="6" fill="#cfe9ff" stroke="#3f7bb2" stroke-width="2.5"/><rect x="360" y="298" width="40" height="12" rx="6" fill="#cfe9ff" stroke="#3f7bb2" stroke-width="2.5"/>'
    '<circle cx="296" cy="354" r="17" fill="#3a4046" stroke="#26303f" stroke-width="3"/><circle cx="296" cy="354" r="8" fill="#8aa7b5"/>'
    '<circle cx="418" cy="354" r="17" fill="#3a4046" stroke="#26303f" stroke-width="3"/><circle cx="418" cy="354" r="8" fill="#8aa7b5"/></g>')
_DRAW["robot"] = lambda rnd, pal: (
    '<g class="sway" style="animation:sway 3.4s ease-in-out infinite;transform-origin:360px 340px">'
    '<rect x="300" y="330" width="120" height="64" rx="12" fill="#cfd8e3" stroke="#a9b7c4" stroke-width="3"/>'
    '<circle cx="360" cy="250" r="52" fill="#cfd8e3" stroke="#a9b7c4" stroke-width="3"/>'
    '<rect x="320" y="210" width="80" height="34" rx="12" fill="#2c3850"/>'
    '<circle cx="342" cy="266" r="12" fill="#7adbe8"/><circle cx="380" cy="266" r="12" fill="#7adbe8"/>'
    '<circle cx="342" cy="266" r="4" fill="#1d2733"/><circle cx="380" cy="266" r="4" fill="#1d2733"/>'
    '<rect x="262" y="360" width="18" height="40" rx="8" fill="#cfd8e3"/><rect x="440" y="360" width="18" height="40" rx="8" fill="#cfd8e3"/>'
    '<circle cx="330" cy="376" r="4" fill="#ff9d45"/><circle cx="360" cy="376" r="4" fill="#ff9d45"/><circle cx="390" cy="376" r="4" fill="#ff9d45"/></g>')
#  —— 鹈鹕经典(精细骑行走单独的 pelican_bike_svg; 这里兜一层通用) ——
#  —— 兜底: 一个随机彩色球体 (认不出主体也不至于白板) ——
_DRAW["__default__"] = lambda rnd, pal: (
    '<g class="bob" style="animation:bob 3s ease-in-out infinite;transform-origin:360px 380px">'
    '<circle cx="360" cy="352" r="44" fill="%s" stroke="#00000022" stroke-width="2"/></g>' % pal["core"])


# ──────────────────────────────────────────────────────────────────────────────
# 通用骨架系统(_KIT): 库里没有的主体, 靠这里把身体拼出来 —— 这才是"生成"不是"查"。
#   骨架函数签名均为 kit(rnd, pal, text); text 用来挂"动作/携带物"道具。
# ──────────────────────────────────────────────────────────────────────────────
def _scene_props(text):
    """从描述里认出「动作/携带物」, 返回要补在画面上的道具 SVG(可为空)。"""
    t = text or ""
    if any(k in t for k in ("编程", "写代码", "敲代码", "程序员", "打字", "敲字", "键盘", "coding")):
        return _prop_laptop()
    if any(k in t for k in ("电脑", "笔记本", "computer")):
        return _prop_laptop()
    if any(k in t for k in ("读书", "看书", "阅读", "写作业", "作业", "书本", "书")):
        return _prop_book()
    if any(k in t for k in ("球", "足球", "篮球", "排球")):
        return _prop_ball()
    return ""


def _prop_laptop():
    """立在 human 的桌面上(坐标已对好 _kit_human 的桌板 y≈366), 键盘有敲击动画。"""
    return ('<g>'
            '<rect x="336" y="300" width="16" height="52" rx="4" fill="#3a4046"/>'
            '<rect x="286" y="296" width="120" height="8" rx="4" fill="#26303f"/>'
            '<rect x="322" y="230" width="76" height="62" rx="5" fill="#8fd0ff" stroke="#4a90c9" stroke-width="3"/>'
            '<path d="M332,244 q7,9 15,6" stroke="#ffffff" fill="none" stroke-width="3" style="animation:flick .8s ease-in-out infinite"/>'
            '<path d="M352,252 q4,6 9,4" stroke="#4a90c9" fill="none" stroke-width="2"/>'
            '<rect x="292" y="300" width="108" height="9" rx="3" fill="#3a4046" style="animation:flick .45s ease-in-out infinite"/>'
            '</g>')


def _prop_book():
    return ('<g><rect x="330" y="300" width="60" height="14" rx="6" fill="#7aa2d6" stroke="#5b8fd6" stroke-width="2"/>'
            '<line x1="360" y1="300" x2="360" y2="314" stroke="#eaf3fc" stroke-width="3"/>'
            '<rect x="300" y="288" width="16" height="8" rx="4" fill="#fff" transform="rotate(-28 300 288)"/></g>')


def _prop_ball():
    return ('<g style="animation:spin360 3s linear infinite">'
            '<circle cx="400" cy="300" r="26" fill="#e4574f" stroke="#c23f38" stroke-width="2.5"/>'
            '<path d="M400,274 q18,26 0,52 q18,-26 0,-52z M400,274 q-18,26 0,52" fill="none" stroke="#c23f38" stroke-width="2"/>'
            '<path d="M374,300 h52" stroke="#c23f38" stroke-width="2"/></g>')


def _kit_human(rnd, pal, text):
    """人: 头 + 身体 + 手臂 + 坐椅; 描述里带「编程/读书」就把电脑/书摆上桌。"""
    skin = rnd.choice(("#f2c9a0", "#e8b88a", "#ffd6b3", "#c68b5e", "#f6d9b8"))
    hair = rnd.choice(("#3a2e2a", "#5b4235", "#1d2733", "#8a5a3a", "#6a4a38"))
    shirt = rnd.choice(("#5b8fd6", "#e4574f", "#57a85c", "#8a6ad6", "#f2a63c"))
    pants = rnd.choice(("#4a5a7a", "#26303f", "#6a4a38", "#444"))
    props = _scene_props(text)
    return (
        '<g class="sway" style="animation:sway 2.8s ease-in-out infinite;transform-origin:360px 390px">'
        f'<circle cx="360" cy="200" r="30" fill="{skin}" stroke="#00000022" stroke-width="2.5"/>'
        f'<path d="M332,190 q-4,-34 26,-40 q30,2 28,40 l-4,6 q-10,-12 -24,-12 q-14,0 -24,8 z" fill="{hair}"/>'
        f'<circle cx="348" cy="198" r="3.6" fill="#1d2733"/><circle cx="372" cy="198" r="3.6" fill="#1d2733"/>'
        f'<path d="M350,214 q10,10 20,0" stroke="#7d4a3a" fill="none" stroke-width="3" stroke-linecap="round"/>'
        f'<ellipse cx="360" cy="282" rx="46" ry="50" fill="{shirt}" stroke="#00000033" stroke-width="2.5"/>'
        f'<path d="M338,246 l8,10 l-6,12 z" fill="#ffffff" opacity=".9"/>'
        f'<path d="M316,252 q-26,22 -18,58 q8,4 12,-10 z" fill="{skin}" stroke="#00000022" stroke-width="2"/>'
        f'<path d="M404,252 q26,22 18,58 q-8,4 -12,-10 z" fill="{skin}" stroke="#00000022" stroke-width="2"/>'
        f'<rect x="278" y="362" width="20" height="26" rx="9" fill="{hair}"/>'  # 椅背
        f'<ellipse cx="360" cy="384" rx="46" ry="20" fill="#8a5a3a" stroke="#00000022" stroke-width="2"/>'
        # 桌板 + 桌腿(挡腿, 坐姿合理)
        '<rect x="250" y="366" width="220" height="14" rx="7" fill="#96643c" stroke="#7d5636" stroke-width="2"/>'
        '<rect x="268" y="380" width="12" height="42" rx="5" fill="#7d5636"/><rect x="440" y="380" width="12" height="42" rx="5" fill="#7d5636"/>'
        f'<rect x="318" y="404" width="26" height="12" rx="6" fill="{pants}"/>'
        f'<rect x="372" y="404" width="26" height="12" rx="6" fill="{pants}"/>'
        + props +
        '</g>')


def _kit_animal(rnd, pal, text):
    """通用四足/直立小动物: 身体 + 头 + 耳 + 眼 + 腿 + 尾巴。"""
    body_c = rnd.choice(("#f2b04c", "#b07a4a", "#c8956a", "#e8b572", "#9a86c2"))
    dark = "#00000033"
    return (
        '<g class="sway" style="animation:sway 3.4s ease-in-out infinite;transform-origin:360px 400px">'
        f'<ellipse cx="360" cy="352" rx="60" ry="34" fill="{body_c}" stroke="{dark}" stroke-width="2.5"/>'
        f'<circle cx="300" cy="300" r="26" fill="{body_c}" stroke="{dark}" stroke-width="2.5"/>'
        '<path d="M282,288 l-8,-24 l14,10 z"/><path d="M310,286 l6,-26 l18,8 z" fill="#e0907a"/>'
        f'<circle cx="292" cy="296" r="3.2" fill="#1d2733"/><circle cx="310" cy="296" r="3.2" fill="#1d2733"/>'
        f'<ellipse cx="300" cy="306" rx="6" ry="4" fill="#e0907a"/>'
        f'<rect x="314" y="382" width="14" height="24" rx="6" fill="{dark}"/><rect x="340" y="382" width="14" height="24" rx="6" fill="{dark}"/>'
        f'<rect x="384" y="382" width="14" height="24" rx="6" fill="{dark}"/><rect x="410" y="382" width="14" height="24" rx="6" fill="{dark}"/>'
        f'<path d="M270,318 q-26,6 -30,-20 q8,6 18,4 q8,-4 18,-2" stroke="{dark}" fill="none" stroke-width="7" stroke-linecap="round"/>'
        '</g>')


def _kit_machine(rnd, pal, text):
    """通用小车: 车身 + 车窗 + 轮子 + 车灯 + 排气管。"""
    body_c = rnd.choice(("#5799d6", "#e4574f", "#57a85c", "#e0b45e", "#8a6ad6"))
    return (
        '<g class="sway" style="animation:sway 4s ease-in-out infinite;transform-origin:360px 350px">'
        f'<path d="M250,352 L252,306 L322,306 L352,282 L436,282 L464,352 z" fill="{body_c}" stroke="#00000033" stroke-width="3"/>'
        f'<path d="M322,306 L350,288 L344,306 z" fill="#00000033"/>'
        '<rect x="298" y="296" width="46" height="14" rx="7" fill="#cfe9ff" stroke="#00000022" stroke-width="2"/>'
        '<rect x="360" y="294" width="44" height="12" rx="6" fill="#cfe9ff" stroke="#00000022" stroke-width="2"/>'
        '<circle cx="330" cy="388" r="8" fill="#ffe27a"/>'
        '<rect x="250" y="346" width="10" height="12" rx="5" fill="#8a8f98"/>'
        '<circle cx="302" cy="388" r="19" fill="#3a4046" stroke="#26303f" stroke-width="3"/><circle cx="302" cy="388" r="8" fill="#8aa7b5"/>'
        '<circle cx="418" cy="388" r="19" fill="#3a4046" stroke="#26303f" stroke-width="3"/><circle cx="418" cy="388" r="8" fill="#8aa7b5"/>'
        '</g>')


def _kit_structure(rnd, pal, text):
    """通用小屋: 墙 + 屋顶 + 门 + 窗 + 烟囱。"""
    wall = rnd.choice(("#f3e6c9", "#e8e3d4", "#ffd9c2", "#e0e7f2"))
    roof = rnd.choice(("#e4574f", "#5b8fd6", "#57a85c", "#8a6ad6"))
    return (
        '<g class="sway" style="animation:sway 5s ease-in-out infinite;transform-origin:360px 360px">'
        f'<path d="M244,330 L360,216 L476,330 z" fill="{roof}" stroke="#00000033" stroke-width="3"/>'
        f'<rect x="256" y="330" width="208" height="72" fill="{wall}" stroke="#00000033" stroke-width="3"/>'
        '<rect x="306" y="352" width="42" height="50" rx="4" fill="#8a5a3a"/>'
        '<circle cx="342" cy="378" r="3" fill="#ffe27a"/>'
        '<rect x="372" y="338" width="26" height="30" rx="8" fill="#7aa2d6" stroke="#5b8fd6" stroke-width="3"/>'
        '<rect x="330" y="338" width="26" height="30" rx="8" fill="#7aa2d6" stroke="#5b8fd6" stroke-width="3"/>'
        '<rect x="300" y="268" width="18" height="62" rx="8" fill="#9aa5b0" stroke="#00000033" stroke-width="2"/>'
        '</g>')


def _kit_generic(rnd, pal, text):
    """万能兜底: 圆滚滚的可爱小角色 —— 任何认不出的描述, 也有个招牌落地。"""
    c = pal["core"]
    body = rnd.choice((c, "#e8b572", "#f2c9a0", "#9ad0e8"))
    return (
        '<g class="sway" style="animation:sway 3s ease-in-out infinite;transform-origin:360px 380px">'
        f'<ellipse cx="360" cy="356" rx="54" ry="44" fill="{body}" stroke="#00000033" stroke-width="2.5"/>'
        f'<circle cx="360" cy="288" r="34" fill="{body}" stroke="#00000033" stroke-width="2.5"/>'
        '<circle cx="346" cy="284" r="4" fill="#1d2733"/><circle cx="374" cy="284" r="4" fill="#1d2733"/>'
        '<path d="M348,302 q12,12 24,0" stroke="#00000044" fill="none" stroke-width="3" stroke-linecap="round"/>'
        '<ellipse cx="330" cy="302" rx="9" ry="6" fill="#e0907a" opacity=".8"/><ellipse cx="390" cy="302" rx="9" ry="6" fill="#e0907a" opacity=".8"/>'
        '<rect x="316" y="396" width="14" height="18" rx="7" fill="#00000022"/><rect x="390" y="396" width="14" height="18" rx="7" fill="#00000022"/>'
        '</g>')


_KIT = {
    "human": _kit_human, "animal": _kit_animal, "machine": _kit_machine,
    "structure": _kit_structure, "generic": _kit_generic,
}


def _backdrop(pal, rnd, weather):
    """天空渐变 + 日/月 + 云 + 海 + 沙 + 浪。返回 (背景, 地平线以上层)。"""
    sky1, sky2, sky3 = pal["sky"][0], pal["sky"][1], pal["sky"][2]
    sea1, sea2 = pal["sea"][0], pal["sea"][1]
    sand1, sand2 = pal["sand"][0], pal["sand"][1]
    is_night = pal["name"] in ("夜色", "月夜") or weather == "moon"
    sun = ('<circle cx="610" cy="120" r="44" fill="%s"/>' % pal["sun"]) if not is_night else \
        ('<circle cx="610" cy="110" r="40" fill="#f2f5ff" opacity=".96"/><circle cx="596" cy="98" r="20" fill="%s" opacity=".85"/>' % pal["sky"][0])
    clouds = "".join(
        '<ellipse cx="%d" cy="%d" rx="%d" ry="%d" fill="#ffffff" opacity=".7"/>'
        % (cx, cy, rx, ry)
        for cx, cy, rx, ry in [(120, 80, 42, 16), (196, 96, 30, 12), (470, 70, 36, 14), (550, 150, 28, 11)])
    return ("""
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="%s"/><stop offset=".6" stop-color="%s"/><stop offset="1" stop-color="%s"/>
    </linearGradient>
    <linearGradient id="sea" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="%s"/><stop offset="1" stop-color="%s"/>
    </linearGradient>
    <linearGradient id="sand" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="%s"/><stop offset="1" stop-color="%s"/>
    </linearGradient>
  </defs>
  <rect width="720" height="440" fill="url(#sky)"/>
  %s
  %s
  <rect x="0" y="250" width="720" height="110" fill="url(#sea)"/>
  %s
  <rect x="0" y="360" width="720" height="80" fill="url(#sand)"/>
""" % (sky1, sky2, sky3, sea1, sea2, sand1, sand2, sun, clouds, _wave_path(266, 1, 14, 8)))


def svgdream_svg(seed=None, info=None, profile=None):
    """按主体 + 气氛生成完整 <svg>。鹈鹕专用回到精细骑行; 其它主体走通用画布。"""
    rnd = random.Random(seed)
    profile = profile or svgdream_intent("一只猫", rnd)
    pal = profile.get("pal_override") or rnd.choice(_PB_PALETTES)
    kind = profile.get("subject") or "cat"
    if kind == "pelican":
        cons = (info or {}).get("cons") or {}
        svg = pelican_bike_svg(seed, info, cons, {
            "weather": profile.get("weather", ""),
            "pal_override": pal, "mood": profile.get("mood", "")})
        return svg
    if kind in _DRAW:
        body = _draw_subject(kind, rnd, pal)
    else:
        # 库里没有 → 用通用骨架现拼; 按动作挂道具(编程→电脑, 看书→书本…)
        kit = kind if kind in _KIT else svgdream_classify(profile.get("raw", ""))
        body = _KIT.get(kit, _kit_generic)(rnd, pal, profile.get("raw", ""))
    bg = _backdrop(pal, rnd, profile.get("weather", ""))
    wet = _weather_overlay(profile)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440" width="720" height="440" role="img" aria-label="%s">'
           '%s%s<g class="pbBob">%s</g>%s</svg>') % (
               profile.get("desc_cn", "画像"), _SVG_STYLE, bg, body, wet)
    return svg


def svgdream_save(svg, folder=None):
    """落盘 .svg, 返回绝对路径; 出任何错都返回空串不抛。"""
    try:
        base = folder or SVGDREAM_SVG_DIR
        os.makedirs(base, exist_ok=True)
        now = datetime.datetime.now()
        name = "画作_%s.svg" % now.strftime("%m%d_%H%M%S")
        path = os.path.join(base, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        return path
    except Exception:
        return ""


def svgdream_save_prompt(prompt, name, folder=None):
    try:
        base = folder or SVGDREAM_SVG_DIR
        os.makedirs(base, exist_ok=True)
        p = os.path.join(base, (name.rsplit(".", 1)[0] if name.endswith(".svg") else name) + ".prompt.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(prompt if prompt.endswith("\n") else prompt + "\n")
        return p
    except Exception:
        return ""


def svgdream_reply(text, seed=None, save=True, folder=None):
    """通用入口的完整回复: 小方先想 → 生成 → 落盘 SVG + Prompt → 回文本。"""
    rnd = random.Random(seed) if seed is not None else random
    profile = svgdream_intent(text, rnd)
    svg = svgdream_svg(seed, None, profile)
    path = svgdream_save(svg, folder) if save else ""
    prompt = profile.get("prompt_en", "") or ""
    ppath = svgdream_save_prompt("小方写好的真图 Prompt(喂给任意 AI 画图就是这幅):\n\n" + prompt + "\n",
                                 os.path.basename(path) if path else "画作", folder) if (save and prompt and path) else ""
    kind = profile.get("subject", "cat")
    _cn_disp = _OBJ_CN.get(kind) or _KIT_META.get(kind, ("这幅小画",))[0]
    lines = ["🎨 小方先想了下你的意思: 你想画%s, %s。" % (profile.get("desc_cn", ""), profile.get("theme", ""))]
    lines.append("")
    lines.append("《%s · %s》画好了:」" % (_cn_disp, profile["palette_name"]))
    lines.append("- 引擎: " + _KIND_ENGINE.get(kind, "通用"))
    lines.append("- 主体: " + _cn_disp)
    lines.append("- 气氛: " + profile.get("theme", "") + ("(" + profile.get("mood", "") + ")" if profile.get("mood") else ""))
    lines.append("- 配色: " + profile["palette_name"])
    lines.append("- 天气: " + (profile.get("weather") or "晴朗"))
    if path:
        lines.append("")
        lines.append("存成了动态 SVG → " + os.path.relpath(path, os.path.dirname(os.path.abspath(__file__))))
    if ppath:
        lines.append("真图 Prompt → " + os.path.relpath(ppath, os.path.dirname(os.path.abspath(__file__))))
    if prompt:
        lines.append("")
        lines.append("✍️ 小方写好的真图 Prompt(复制喂给任何 AI 画图就是这幅):")
        lines.append(prompt)
    if not path and not ppath:
        lines.append("")
        lines.append("（落盘没成功, 但画面已经生成好了）")
    return "\n".join(lines)


def svgdream_answer(text):
    """通用统一入口: 是「画/写/生成 XXX」就出图, 否则 None 让位别的路由。"""
    try:
        if svgdream_is_request(text):
            return svgdream_reply(text, folder=SVGDREAM_SVG_DIR)
    except Exception:
        return None
    return None
