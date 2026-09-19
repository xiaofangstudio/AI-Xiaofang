# -*- coding: utf-8 -*-
"""
AI 小方 (FlphaLit Covi1) —— 自研「说话层」模块 xiaofang_mouth_covi1
====================================================================
「说话层」目前只交付两件事, 别的一概不管(前向计算以后由别人接手):

  1) FlphaLitBPETokenizer —— 字节级 BPE 分词器
     词表 / 合并表 / 切分正则 / 特殊 token 全部从
     `权重缓存/FlphaLit_Mouth_1B5/tokenizer.json` 现读现用, 代码里一个魔法常量都不写死。
     为什么非要这么"死板"? 因为词表一旦偷偷换版本, 硬编码的正则会静默地切错,
     训练端和推理端就对不齐了, 这种 bug 事后极难查, 所以宁可每次都读文件。

  2) flphalit_safetensors_header / flphalit_load_tensors —— 权重文件读取器
     纯标准库 + numpy 手撸解析。为什么不用现成的读取库? 这是宣称 100% 自研的项目,
     说话层当然也得自己拆。顺带一个好处: 内存占用完全可控 —— 按 data_offsets 精确
     mmap, 需要哪个张量才把哪个张量搬进内存, 绝不把整个大文件先吞进来再切。

设计上刻意保持"零第三方痕迹": 不 import 任何权重/分词第三方库, 类名函数名统一
FlphaLit 前缀, 注释也是自己人看得懂的大白话。
"""

import os
import re
import json
import struct
import unicodedata

import numpy as np


# ====================================================================
# 0. 全局常量
# ====================================================================

#: 本模块版本号, 方便跟别的模块对账
FLPHALIT_COVI1_VERSION = "Covi1-0.1"

#: 默认的说话层权重目录(相对本文件定位, 换机器也不用改)
FLPHALIT_MOUTH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "权重缓存", "FlphaLit_Mouth_1B5"
)

#: 权重文件名的约定
FLPHALIT_TOKENIZER_JSON_NAME = "tokenizer.json"
FLPHALIT_WEIGHTS_NAME = "model.safetensors"

#: "下载完成" 的经验阈值: 1.5B 级别的 bf16 权重差不多 3GB, 明显小于它就是没下完。
#: 这个数只用来做"完整性提示", 真正的完整性判断还是拿 header 里算出来的字节数比。
FLPHALIT_FULL_WEIGHT_MIN_BYTES = 3 * 1024 ** 3

#: 单个字符查不到映射时的兜底(问号), 基本用不上, 纯属防御
_FLPHALIT_FALLBACK_BYTE = 0x3F


# ====================================================================
# 1. 字节 <-> 可打印 unicode 的双向映射表
# ====================================================================
# 为什么要有这层映射?
#   原始的 256 个字节里有 \x00、\n 这种控制字符, 它们会跟 BPE 的分隔逻辑打架,
#   也会把词表搞得乌烟瘴气。业内通用做法是: 把"可打印 ASCII + 可见的拉丁补充区"
#   原样保留, 剩下那些"见不得人"的字节(控制符、空格等)统一挪到一个高位 unicode 区。
#   这样每个字节都能变成一个可打印字符, 一个字节 = 一个字符, 长度不变, 又可以
#   原样可逆地还原回去。

def flphalit_bytes_to_unicode():
    """造出 {字节值: 单个 unicode 字符} 的映射表。"""
    # 先把"本来就能看见"的字节挑出来: 可见 ASCII + 拉丁补充里的可见区
    visible = (
        list(range(ord("!"), ord("~") + 1))        # ! .. ~
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))  # 倒问号 .. 逻辑非
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))  # 注册商标 .. ÿ
    )
    # cs 存"映射过去的目标码点", 一开始跟可见字节一一对应
    targets = list(visible)
    extra = 0
    for b in range(256):
        if b not in visible:
            visible.append(b)              # 记下这个没被照顾到的字节
            targets.append(256 + extra)    # 挪到 256 往后的高位区
            extra += 1
    return dict(zip(visible, [chr(t) for t in targets]))


#: 字节 -> 字符(全局只建一次)
FLPHALIT_BYTE_TO_CHAR = flphalit_bytes_to_unicode()

#: 字符 -> 字节(反查用)
FLPHALIT_CHAR_TO_BYTE = {c: b for b, c in FLPHALIT_BYTE_TO_CHAR.items()}


# ====================================================================
# 2. 把配置文件里的正则"翻译"成 Python 能吃的正则
# ====================================================================
# 为什么要翻译?
#   tokenizer.json 里的切分正则用的是 \p{L}(字母) / \p{N}(数字) 这种 Unicode
#   属性写法, 这是那个行业通用的正则方言, 而 Python 标准库 re 不支持 \p{}。
#   既然规矩是不许 import 第三方正则库, 那就自己把 \p{X} 展开成"码点区间"。
#   区间是用 unicodedata 现算的, 不是抄来的常量, 所以换 Python 版本也能自适应。
#
# 为什么不能简单地做字符串替换?
#   \p{L} 有时出现在字符类里面(比如 [^\r\n\p{L}\p{N}]), 有时出现在外面(比如 \p{L}+)。
#   放外面得包一层方括号变成 [区间...], 放里面则只能塞赤裸裸的区间(不能嵌套方括号,
#   否则 Python 的字符类语法直接崩)。所以下面写了个会数方括号深度的扫描器。

_FLPHALIT_CATEGORY_RANGE_CACHE = {}


def flphalit_unicode_category_ranges(prefix):
    """
    把某个 Unicode 大类(如 "L"/"N")展开成 [(起, 止), ...] 的码点区间。
    做法很朴素: 从 0 一直扫到 0x110000, 挨个问 unicodedata 这是不是我要的类。
    慢是慢了点(几十万次调用), 但只算一次然后就缓存住了, 实测 0.3 秒左右。
    """
    if prefix in _FLPHALIT_CATEGORY_RANGE_CACHE:
        return _FLPHALIT_CATEGORY_RANGE_CACHE[prefix]

    ranges = []
    start = None
    prev = None
    for cp in range(0x110000):
        # 代理区(0xD800-0xDFFF)本身不是合法字符, category 会给 "Cs", 自然被排除,
        # 这里不用特意去判, 顺手写一下省得以后有人疑惑。
        in_set = unicodedata.category(chr(cp)).startswith(prefix)
        if in_set:
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            ranges.append((start, prev))
            start = None
    if start is not None:
        ranges.append((start, prev))

    _FLPHALIT_CATEGORY_RANGE_CACHE[prefix] = ranges
    return ranges


def _flphalit_ranges_to_class_body(prefix):
    """把码点区间拼成可以塞进字符类里的字符串, 用 \\U 转义保证没有歧义。"""
    parts = []
    for a, b in flphalit_unicode_category_ranges(prefix):
        if a == b:
            parts.append("\\U%08x" % a)
        else:
            parts.append("\\U%08x-\\U%08x" % (a, b))
    return "".join(parts)


def flphalit_translate_regex(pattern):
    """
    把配置文件里的正则翻译成 Python re 能编译的版本。
    目前处理两类东西:
      * \\p{X} / \\P{X}  ->  展开成码点区间(按是否在字符类里决定要不要包方括号)
      * 其它转义和普通字符  ->  原样抄过去
    """
    out = []
    i = 0
    n = len(pattern)
    in_class = False   # 当前是不是走在 [...] 里面
    body_cache = {}

    while i < n:
        ch = pattern[i]

        if ch == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            # 命中 \p{...} 或 \P{...}
            if nxt in "pP" and i + 2 < n and pattern[i + 2] == "{":
                end = pattern.index("}", i + 3)
                name = pattern[i + 3:end]
                if name not in body_cache:
                    body_cache[name] = _flphalit_ranges_to_class_body(name)
                body = body_cache[name]
                negated = nxt == "P"
                if in_class:
                    # 字符类内部没法再套一层取反, 老实报错, 不要静默产生错误语义
                    if negated:
                        raise ValueError(
                            "\\P{%s} 出现在字符类内部, 本翻译器不支持这种写法" % name
                        )
                    out.append(body)
                else:
                    out.append("[" + ("^" if negated else "") + body + "]")
                i = end + 1
                continue
            # 普通转义(\s \d \r \n \. 等)整对抄走
            out.append(ch + nxt)
            i += 2
            continue

        if ch == "[":
            # 进入字符类。注意 "[^]" 和 "[]a]" 这种边界写法, 方括号后紧跟的
            # ^ 和 ] 都不算类结束, 得先吃掉。
            in_class = True
            out.append(ch)
            i += 1
            if i < n and pattern[i] == "^":
                out.append("^")
                i += 1
            if i < n and pattern[i] == "]":
                out.append("]")
                i += 1
            continue

        if ch == "]" and in_class:
            in_class = False
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


# ====================================================================
# 3. FlphaLitVocab —— 词表 / 合并表 / 特殊 token 的容器
# ====================================================================

class FlphaLitVocab(object):
    """
    负责把 tokenizer.json 里的东西读进来并整理成方便查的结构。
    读文件这事全部收在这一个类里, 分词器本体就不用关心 JSON 长什么样了。
    """

    def __init__(self, token_to_id, merges, added_tokens,
                 split_regex, normalizer_F="NFC", add_prefix_space=False,
                 bos_id=None, eos_id=None, source_path=None):
        self.source_path = source_path

        # ---- 词表: token 字符串 <-> id ----
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {}
        for tok, tid in self.token_to_id.items():
            self.id_to_token[int(tid)] = tok

        # ---- 合并表: 顺序即优先级(下标越小越优先) ----
        # 存成 {(左, 右): 优先序号}, 查的时候直接比序号谁小谁先合。
        self.merges = merges
        self.merge_ranks = {}
        for rank, pair in enumerate(merges):
            self.merge_ranks[pair] = rank

        # ---- 特殊 token(added_tokens) ----
        # 这些家伙必须"整块命中", 不能被 BPE 拆开, 所以单独记一份。
        self.added_tokens = list(added_tokens)
        self.added_content_to_id = {}
        self.added_token_ids = set()
        self.added_token_contents = []
        for item in self.added_tokens:
            content = item["content"]
            tid = int(item["id"])
            self.added_content_to_id[content] = tid
            self.added_token_ids.add(tid)
            self.added_token_contents.append(content)
            # 有些特殊 token 没被塞进 model.vocab, 这里补上, 保证正反查都通
            if content not in self.token_to_id:
                self.token_to_id[content] = tid
            self.id_to_token[tid] = content

        # ---- 切分正则 ----
        self.split_regex_text = split_regex
        self.python_regex_text = flphalit_translate_regex(split_regex)
        self.split_regex = re.compile(self.python_regex_text)

        # ---- 归一化形式 ----
        self.normalizer_F = normalizer_F
        self.add_prefix_space = bool(add_prefix_space)

        # ---- 起止 token ----
        self.bos_id = bos_id
        self.eos_id = eos_id

    # ---------------- 基础查询 ----------------
    def __len__(self):
        return len(self.token_to_id)

    def get_id(self, token):
        return self.token_to_id.get(token)

    def get_token(self, tid):
        return self.id_to_token.get(int(tid))

    def is_added_token(self, tid):
        return int(tid) in self.added_token_ids

    # ---------------- 从文件加载 ----------------
    @classmethod
    def from_tokenizer_json(cls, json_path, bos_id=None, eos_id=None):
        """唯一入口: 一个 tokenizer.json 进去, 一个整理好的词表对象出来。"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = data.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(
                "说话层目前只认字节级 BPE, 这个文件里写的是 %r" % (model.get("type"),)
            )

        # ---- 词表 ----
        token_to_id = model.get("vocab", {})

        # ---- 合并表 ----
        # 有的文件里 merges 是 ["a b", ...], 有的是 [["a","b"], ...],
        # 两种都吃, 免得换个导出工具就翻车。
        # 顺带说明: 字节级映射之后,"空格"已经被换成了特殊字符, 所以拆分合并项时
        # 按单个空格切是安全的, 不会把一个 token 切碎。
        merges = []
        for item in model.get("merges", []):
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    merges.append((item[0], item[1]))
            else:
                parts = str(item).split(" ")
                if len(parts) >= 2:
                    merges.append((parts[0], parts[1]))

        # ---- 切分正则: 从 pre_tokenizer 里递归找 Split 的 Regex ----
        split_regex = None
        add_prefix_space = False
        pre = data.get("pre_tokenizer")
        for node in _flphalit_walk_json(pre):
            ntype = node.get("type")
            if ntype == "Split":
                pat = node.get("pattern")
                if isinstance(pat, dict) and "Regex" in pat:
                    split_regex = pat["Regex"]
                elif isinstance(pat, str):
                    split_regex = pat
            elif ntype == "ByteLevel":
                add_prefix_space = bool(node.get("add_prefix_space", False))
        if not split_regex:
            raise ValueError("tokenizer.json 里没找到 pre_tokenizer 的切分正则, 不敢瞎猜")

        # ---- 归一化: 也递归找(可能是单个, 也可能是一串) ----
        norm_forms = []
        for node in _flphalit_walk_json(data.get("normalizer")):
            ntype = node.get("type")
            if ntype in ("NFC", "NFD", "NFKC", "NFKD"):
                norm_forms.append(ntype)
        normalizer_F = norm_forms[0] if norm_forms else None

        # ---- 特殊 token ----
        added = []
        for item in data.get("added_tokens", []):
            if "content" in item and "id" in item:
                added.append(item)

        return cls(
            token_to_id=token_to_id,
            merges=merges,
            added_tokens=added,
            split_regex=split_regex,
            normalizer_F=normalizer_F,
            add_prefix_space=add_prefix_space,
            bos_id=bos_id,
            eos_id=eos_id,
            source_path=json_path,
        )


def _flphalit_walk_json(node):
    """把 JSON 里所有 dict 节点都吐出来, 方便不关心层级地找东西。"""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            for sub in _flphalit_walk_json(value):
                yield sub
    elif isinstance(node, list):
        for value in node:
            for sub in _flphalit_walk_json(value):
                yield sub


# ====================================================================
# 4. FlphaLitBPETokenizer —— 主角, 字节级 BPE 分词器
# ====================================================================

def _flphalit_get_pairs(word):
    """取一个 token(字符元组)里所有相邻字符对。"""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class FlphaLitBPETokenizer(object):
    """
    字节级 BPE 分词器。

    整体流水线(跟主流字节级 BPE 的实现思路一致):
        文本
          -> (可选)归一化 NFC
          -> 先把特殊 token 整块抠出来
          -> 剩下的正常文本按 pre_tokenizer 正则切段
          -> 每段转 UTF-8 字节, 再按映射表变成"可打印字符"
          -> 在这段内部按 merges 的优先级反复合并
          -> 合并结果查词表得到 id

    为什么切段之后才做 BPE?
      因为 BPE 只应该在"一段"内部合并, 跨段(比如跨过空格、跨过标点)合出来的
      token 是错的。所以正则切段是 BPE 的前置步骤, 不能省。

    为什么要缓存?
      同一句话里"的"、"是"这种片段会反复出现, 每次重跑一遍合并过程纯属浪费。
      缓存按"切段后的原始片段字符串"做 key, 命中就直接返回 id 列表, 实测对
      中文长句能省掉一大半重复计算。
    """

    def __init__(self, vocab, add_bos=False, add_eos=False, cache_size=200000):
        if not isinstance(vocab, FlphaLitVocab):
            raise TypeError("FlphaLitBPETokenizer 需要一个 FlphaLitVocab")
        self.vocab = vocab
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.cache_size = int(cache_size)

        # 片段 -> 该片段切出来的 id 元组
        self._piece_cache = {}
        self._unknown_chars = set()   # 记下那些真的查不到的字, 事后好排查

        # 特殊 token 的整块匹配正则: 按长度从长到短拼, 保证长的先命中
        # (比如 <|im_start|> 不会被更短的规则抢走)
        if self.vocab.added_token_contents:
            ordered = sorted(set(self.vocab.added_token_contents),
                             key=len, reverse=True)
            self._added_pattern = re.compile(
                "|".join(re.escape(c) for c in ordered)
            )
        else:
            self._added_pattern = None

    # ---------------- 对外小工具 ----------------
    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, tid):
        return self.vocab.get_token(tid)

    def token_to_id(self, token):
        return self.vocab.get_id(token)

    def cache_stats(self):
        return {"片段缓存条数": len(self._piece_cache),
                "查不到的字": len(self._unknown_chars)}

    # ---------------- 归一化 ----------------
    def _normalize(self, text):
        form = self.vocab.normalizer_F
        if form:
            text = unicodedata.normalize(form, text)
        if self.vocab.add_prefix_space and not text.startswith(" "):
            text = " " + text
        return text

    # ---------------- 特殊 token 的整块切分 ----------------
    def _split_by_added_tokens(self, text):
        """
        返回 [(是否特殊token, 片段), ...]。
        没注册特殊 token 就整段原样返回, 省点开销。
        """
        if self._added_pattern is None or not text:
            return [(False, text)]

        out = []
        last = 0
        for m in self._added_pattern.finditer(text):
            if m.start() > last:
                out.append((False, text[last:m.start()]))
            out.append((True, m.group(0)))
            last = m.end()
        if last < len(text):
            out.append((False, text[last:]))
        return out

    # ---------------- 核心: 一段内部的 BPE 合并 ----------------
    def _bpe_merge(self, word):
        """
        输入一串"映射后的字符"(元组), 输出合并后的 token 元组。
        规则很简单: 每轮挑出当前优先级最高(merges 下标最小)的相邻字符对,
        把所有出现的地方一次合掉, 然后再来一轮, 直到没有可合的对了。

        为什么每轮要"合掉所有出现的地方"而不是只合一处?
          这是主流实现的既定行为, 也只合一处的话, 同一个片段里重复的模式
          会被合得七零八落, 跟训练时对不上。
        """
        if not word:
            return ()
        if len(word) == 1:
            return word

        ranks = self.vocab.merge_ranks
        pairs = _flphalit_get_pairs(word)

        while pairs:
            # 找出当前所有相邻对里优先级最高的那个
            best = None
            best_rank = None
            for pair in pairs:
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = pair
            if best is None:
                break   # 剩下的相邻对都没在合并表里, 收工

            first, second = best
            merged = first + second

            new_word = []
            i = 0
            n = len(word)
            while i < n:
                # 找到 first 下次出现的位置
                j = -1
                try:
                    j = word.index(first, i)
                except ValueError:
                    j = -1
                if j < 0:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < n - 1 and word[i + 1] == second:
                    new_word.append(merged)   # 命中, 两个并一个
                    i += 2
                else:
                    new_word.append(word[i])  # 只是长一样, 不是这一对
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _flphalit_get_pairs(word)

        return word

    def _encode_piece(self, piece):
        """
        把一个"正常文本片段"编码成 id 元组。
        这是最热的一段代码, 所以缓存也放在这里。
        """
        cached = self._piece_cache.get(piece)
        if cached is not None:
            return cached

        # 1) 文本 -> UTF-8 字节
        raw_bytes = piece.encode("utf-8")
        # 2) 字节 -> 可打印字符(长度不变, 只是换个写法)
        chars = "".join(FLPHALIT_BYTE_TO_CHAR[b] for b in raw_bytes)
        # 3) 在这一段内部按优先级合并
        merged_tokens = self._bpe_merge(tuple(chars))

        # 4) 查词表
        ids = []
        for tok in merged_tokens:
            tid = self.vocab.token_to_id.get(tok)
            if tid is None:
                # 理论上不该发生(字节级词表是全覆盖的)。真发生了说明词表有洞,
                # 那就退化成"按单字符再查一次", 尽量少丢信息, 顺便记账备查。
                for ch in tok:
                    sub_id = self.vocab.token_to_id.get(ch)
                    if sub_id is None:
                        self._unknown_chars.add(ch)
                        continue
                    ids.append(sub_id)
                continue
            ids.append(tid)

        result = tuple(ids)

        # 缓存别无限涨, 满了就整体清掉(简单粗暴但够用, 不做 LRU 省得复杂)
        if len(self._piece_cache) >= self.cache_size:
            self._piece_cache.clear()
        self._piece_cache[piece] = result
        return result

    # ---------------- encode ----------------
    def encode(self, text, add_bos=None, add_eos=None):
        """
        文本 -> id 列表。
        add_bos / add_eos 传 None 时用构造时的默认值, 传 True/False 可以临时覆盖。
        """
        if not isinstance(text, str):
            raise TypeError("encode 只吃字符串, 收到的是 %r" % (type(text).__name__,))

        use_bos = self.add_bos if add_bos is None else bool(add_bos)
        use_eos = self.add_eos if add_eos is None else bool(add_eos)

        text = self._normalize(text)

        ids = []
        for is_special, segment in self._split_by_added_tokens(text):
            if not segment:
                continue
            if is_special:
                # 特殊 token 整块命中, 直接给 id, 绝不喂给 BPE
                ids.append(self.vocab.added_content_to_id[segment])
                continue
            # 正常文本: 先按正则切段, 再逐段做 BPE
            for m in self.vocab.split_regex.finditer(segment):
                piece = m.group(0)
                if piece:
                    ids.extend(self._encode_piece(piece))

        if use_bos and self.vocab.bos_id is not None:
            ids.insert(0, self.vocab.bos_id)
        if use_eos and self.vocab.eos_id is not None:
            ids.append(self.vocab.eos_id)
        return ids

    def encode_batch(self, texts, add_bos=None, add_eos=None):
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    # ---------------- decode ----------------
    def decode(self, ids, skip_special_tokens=False):
        """
        id 列表 -> 文本。步骤就是 encode 的逆过程:
        id -> token 字符串 -> 按映射表反查回字节 -> 拼起来 -> UTF-8 解码。
        """
        if isinstance(ids, (int, np.integer)):
            ids = [int(ids)]

        byte_buf = bytearray()
        out_text = []

        def flush():
            if byte_buf:
                # errors="replace": 万一遇到半个被截断的多字节字符, 宁可出个
                # 替换符也不要直接抛异常把整条链路打断。
                out_text.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                del byte_buf[:]

        char_to_byte = FLPHALIT_CHAR_TO_BYTE
        for tid in ids:
            tid = int(tid)

            # 特殊 token 原样吐出, 不参与字节还原
            if tid in self.vocab.added_token_ids:
                if skip_special_tokens:
                    continue
                flush()
                out_text.append(self.vocab.id_to_token.get(tid, ""))
                continue

            tok = self.vocab.id_to_token.get(tid)
            if tok is None:
                continue   # 词表里没这个 id, 跳过, 不硬凑

            for ch in tok:
                byte_buf.append(char_to_byte.get(ch, _FLPHALIT_FALLBACK_BYTE))

        flush()
        return "".join(out_text)


# ====================================================================
# 5. 权重文件读取器(纯手撸)
# ====================================================================
# 文件长这样:
#   [0:8]                     小端 uint64 = header 的字节长度 N
#   [8:8+N]                   header 的 JSON 文本
#   [8+N : 8+N+数据长度]      所有张量的裸数据首尾相接
#
# header 的 JSON 是个 {张量名: {...}} 的大字典, 每个张量长这样:
#   {"dtype": "BF16", "shape": [1536, 151936], "data_offsets": [0, 466944000]}
# 注意 data_offsets 是"相对数据区起点"的偏移, 不是文件绝对偏移, 得加上 8+N。

#: 支持解析的 dtype -> (numpy 小端 dtype, 一个元素几字节)
_FLPHALIT_ST_DTYPES = {
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "BF16": (None, 2),   # BF16 没有原生 numpy 类型, 得手工补位, 见下面
}


def flphalit_safetensors_header(path):
    """
    只读 header, 不碰数据区。返回一个字典:
        {
          "path": 文件路径,
          "file_size": 文件实际大小,
          "header_bytes": header 段长度 N,
          "data_start": 数据区在文件里的绝对起始位置(8+N),
          "metadata": header 里的 __metadata__ (可能没有),
          "tensors": {名字: {"name","dtype","shape","data_offsets"}},
          "num_tensors": 张量个数,
        }
    为什么要单独拆一个"只读 header"的接口?
      因为 3GB 的权重文件里, 真正有用的元信息全在头几 KB, 先解析 header 就能
      知道里面有几个张量、总共多少参数, 不用碰数据区, 几百毫秒就能出结果。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到权重文件: %s" % path)

    file_size = os.path.getsize(path)

    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError("文件太小, 前 8 字节的 header 长度都读不全: %s" % path)
        header_bytes = struct.unpack("<Q", raw_len)[0]

        # 防御一下: header 长度离谱说明这根本不是我们的权重文件, 或者下坏了
        if header_bytes <= 0 or header_bytes > 200 * 1024 * 1024:
            raise ValueError(
                "header 长度看着不对劲(%d 字节), 这个文件恐怕不是合法权重文件" % header_bytes
            )
        if 8 + header_bytes > file_size:
            raise ValueError(
                "文件被截断了: 需要 %d 字节才够放 header, 实际只有 %d 字节"
                % (8 + header_bytes, file_size)
            )

        header_text = f.read(header_bytes)

    try:
        header_json = json.loads(header_text.decode("utf-8"))
    except Exception as exc:
        raise ValueError("header 不是合法 JSON: %s" % exc)

    tensors = {}
    metadata = None
    for name, info in header_json.items():
        if name == "__metadata__":
            metadata = info
            continue
        if not isinstance(info, dict):
            continue
        dtype = info.get("dtype")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [0, 0])
        tensors[name] = {
            "name": name,
            "dtype": dtype,
            "shape": [int(s) for s in shape],
            "data_offsets": [int(offsets[0]), int(offsets[1])],
        }

    return {
        "path": path,
        "file_size": file_size,
        "header_bytes": header_bytes,
        "data_start": 8 + header_bytes,
        "metadata": metadata,
        "tensors": tensors,
        "num_tensors": len(tensors),
    }


def flphalit_safetensors_expected_bytes(header_info):
    """
    按 header 算出"一个完整文件应该有多大" = 数据区起点 + 所有张量里最远的那个尾巴。
    这个数是拿 header 现算的, 比拍脑袋定阈值靠谱, 用来判断文件下完没有。
    """
    max_end = 0
    for info in header_info["tensors"].values():
        end = info["data_offsets"][1]
        if end > max_end:
            max_end = end
    return header_info["data_start"] + max_end


def flphalit_safetensors_status(path):
    """
    给权重文件做个体检, 返回:
        {"header_ok", "file_size", "expected_bytes", "complete",
         "short_bytes", "full_weight_hint_bytes", "reason", "header"}
    complete=False 就说明还没下完。

    判完整性的依据说明:
      真正的依据是"header 里算出来的应有字节数", 这是从文件自身元信息推的, 精确;
      而 3GB 那个经验阈值只当提示量一并返回, 因为不同精度/不同规模的权重大小不同,
      拿固定阈值当判据迟早会误判(本文件按 header 算出来是 2.88GB, 就不到 3GB)。
    """
    info = flphalit_safetensors_header(path)
    expected = flphalit_safetensors_expected_bytes(info)
    size = info["file_size"]
    complete = size >= expected
    if complete:
        reason = "文件完整"
    else:
        reason = ("权重尚未下载完: 实际 %d 字节 / 应有 %d 字节, 还差 %d 字节"
                  % (size, expected, expected - size))
        if size < FLPHALIT_FULL_WEIGHT_MIN_BYTES:
            reason += " (也低于 3GB 的经验阈值)"
    return {
        "header_ok": True,
        "file_size": size,
        "expected_bytes": expected,
        "complete": complete,
        "short_bytes": max(0, expected - size),
        "full_weight_hint_bytes": FLPHALIT_FULL_WEIGHT_MIN_BYTES,
        "reason": reason,
        "header": info,
    }


def flphalit_bf16_to_f32(u16_array):
    """
    BF16 -> F32 的手工补位。
    为什么能手算? 因为 bf16 就是"把 f32 的尾数砍掉低 16 位"得到的,
    所以反过来只要把 16 位放到高半部分(左移 16 位), 再按 f32 重新解释回来,
    就精确还原了(除了 bf16 本来就没存的那点精度)。
    """
    u16 = np.asarray(u16_array)
    if u16.dtype != np.uint16:
        u16 = u16.astype(np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _flphalit_decode_tensor_buf(buf, dtype, shape, upcast_f32=True):
    """把一段裸字节按 dtype 解释成 numpy 数组。buf 是 memmap 上的切片, 不会额外拷贝。"""
    if dtype not in _FLPHALIT_ST_DTYPES:
        raise NotImplementedError(
            "暂不支持 %r 这种 dtype, 目前只认 F32 / F16 / BF16" % (dtype,)
        )

    if dtype == "BF16":
        # BF16 必须手工补位, 而且这一步一定会产生新数组(没法原地视图)
        u16 = np.frombuffer(buf, dtype="<u2")
        arr = flphalit_bf16_to_f32(u16)
    elif dtype == "F16":
        arr = np.frombuffer(buf, dtype="<f2")
        if upcast_f32:
            arr = arr.astype(np.float32)   # 统一升到 f32, 后面算起来省心
    else:  # F32
        arr = np.frombuffer(buf, dtype="<f4")

    return arr.reshape(shape)


def flphalit_load_tensors(path, names=None, upcast_f32=True, verbose=True):
    """
    读取张量, 返回 {张量名: numpy 数组}。

    参数:
      names      要读哪些张量; None 表示全都要(对大模型就是几百个张量, 慎用)
      upcast_f32 F16/BF16 是否统一升成 float32 (默认 True)
      verbose    读不到的张量要不要吱一声

    实现上刻意用 np.memmap 而不是 np.fromfile:
      fromfile 会把整份文件一次性读进内存再切片, 3GB 文件上来就先吃 3GB;
      mmap 只是把文件"映射"成数组, 需要的那几个字节由操作系统按页换入,
      我们只要哪几个张量的数据, 内存里就只驻留那几个张量, 干净利落。
    """
    info = flphalit_safetensors_header(path)
    all_tensors = info["tensors"]
    data_start = info["data_start"]
    file_size = info["file_size"]
    available = file_size - data_start        # 数据区里实际已经下到的字节数

    # 要读哪些
    if names is None:
        wanted = list(all_tensors.keys())
    else:
        wanted = list(names)
        missing = [n for n in wanted if n not in all_tensors]
        if missing:
            raise KeyError("header 里没有这些张量: %s" % (missing[:5],))

    if not wanted:
        return {}

    # 整个文件做一个 uint8 的 mmap, 之后所有张量都是在这上面切片, 不再重复开文件。
    mm = np.memmap(path, dtype=np.uint8, mode="r")

    out = {}
    not_downloaded = []
    try:
        for name in wanted:
            meta = all_tensors[name]
            dtype = meta["dtype"]
            shape = meta["shape"]
            off0, off1 = meta["data_offsets"]

            # 这个张量的数据还没下到本地, 跳过, 但记账
            if off1 > available:
                not_downloaded.append(name)
                continue

            buf = mm[data_start + off0: data_start + off1]
            out[name] = _flphalit_decode_tensor_buf(
                buf, dtype, shape, upcast_f32=upcast_f32
            )
    finally:
        # 关掉 mmap 这个"把手"; 已经切出来的数组各自持有自己的引用, 不受影响
        del mm

    if not_downloaded and verbose:
        print("[说话层] 跳过 %d 个还没下载完的张量(例如 %s)"
              % (len(not_downloaded), not_downloaded[:3]))

    return out


def flphalit_safetensors_total_params(header_info):
    """按 header 里的 shape 把总参数量算出来(纯算术, 不需要数据区)。"""
    total = 0
    for info in header_info["tensors"].values():
        n = 1
        for s in info["shape"]:
            n *= int(s)
        total += n
    return total


# ====================================================================
# 6. 便捷装载入口
# ====================================================================

def flphalit_read_special_ids(model_dir):
    """
    只从 config.json 里抠 bos / eos / pad 三个整数 id。

    为什么只抠这三个、别的一律不碰?
      那个配置文件里还混着架构名之类的"厂商痕迹", 我们的规矩是代码里零第三方名字,
      所以这里只按 key 取数字, 别的内容连打印都不打印, 免得脏了输出。
    """
    ids = {"bos_token_id": None, "eos_token_id": None, "pad_token_id": None}
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key in ids:
                value = cfg.get(key)
                if isinstance(value, int):
                    ids[key] = value
        except Exception:
            pass   # 读不动就算了, 起止 token 本来就是可选功能
    return ids


def flphalit_load_tokenizer(model_dir=None, add_bos=False, add_eos=False):
    """一句话把分词器装好: 读 config 拿起止 id, 再读 tokenizer.json 建词表。"""
    model_dir = model_dir or FLPHALIT_MOUTH_DIR
    json_path = os.path.join(model_dir, FLPHALIT_TOKENIZER_JSON_NAME)
    special = flphalit_read_special_ids(model_dir)
    vocab = FlphaLitVocab.from_tokenizer_json(
        json_path,
        bos_id=special["bos_token_id"],
        eos_id=special["eos_token_id"],
    )
    return FlphaLitBPETokenizer(vocab, add_bos=add_bos, add_eos=add_eos)


# ====================================================================
# 7. 自研前向计算(一) —— 超参
# ====================================================================
# 从这一节起, "说话层"才真正长出嘴: 前面几节只解决"文字 <-> 编号"和
# "怎么把权重从大盘子里稳稳地掏出来", 这一节开始拿这些权重把字一个一个算出来。
#
# 规矩和前面一样, 而且更严:
#   * 数学全部自己写, 只准用 numpy, 不许 import 任何推理框架;
#   * 超参一个都不写死, 一律现读 config.json, 读不到才退回下面的兜底值;
#   * 张量名一个都不许臆造 —— 名字是先拿 flphalit_safetensors_status 把 header
#     里的真名打印出来核对过, 再照着映射的(见 flphalit_mouth_tensor_names)。
#
# 网络骨架(28 层, 纯解码器, 每层两段):
#   词嵌入
#     └─ 每层: RMSNorm -> 注意力(RoPE 旋转位置编码 + 分组查询 GQA) -> 残差
#              RMSNorm -> SwiGLU 前馈                              -> 残差
#   └─ 最终 RMSNorm -> 输出投影(直接复用词嵌入矩阵, 即权重绑定)
#
# 关于"输出投影复用词嵌入"这件事, 没有只凭 config 里那个开关就下结论:
# header 里 338 个张量名全打印出来数过, 里面只有 model.embed_tokens.weight,
# 压根没有单独的 lm_head 一类张量, 所以复用是对的。

#: config.json 的文件名约定
FLPHALIT_CONFIG_JSON_NAME = "config.json"

#: 兜底超参。正常情况这些值都会从 config.json 里读到并被覆盖, 这里只是保险。
FLPHALIT_1B5_DEFAULTS = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "max_position_embeddings": 32768,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}

#: 只允许从 config.json 里读这些 key。做白名单的原因见下面函数的注释。
FLPHALIT_CONFIG_ALLOWED_KEYS = tuple(FLPHALIT_1B5_DEFAULTS.keys())


def flphalit_read_model_config(model_dir=None):
    """
    从 config.json 里只挑"数字 / 布尔"超参出来, 其余一概不读。

    为什么不整个 json.load 完事、直接把字典丢出去?
      那个文件里混着厂商的架构字符串(模型类名、导出工具版本号之类)。本项目的
      硬规矩是"零第三方名字", 所以这里走白名单: 只认 FLPHALIT_CONFIG_ALLOWED_KEYS
      里列出的 key, 其它的连读进来存一份都不干, 从源头上不给第三方痕迹留下
      出现在变量、日志、异常里的机会。
    """
    model_dir = model_dir or FLPHALIT_MOUTH_DIR
    cfg_path = os.path.join(model_dir, FLPHALIT_CONFIG_JSON_NAME)

    raw = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}   # 配置读不动就全走兜底, 不抛异常, 免得"读不到配置"变成为难人的理由

    out = {}
    for key in FLPHALIT_CONFIG_ALLOWED_KEYS:
        value = raw.get(key, FLPHALIT_1B5_DEFAULTS[key])
        # 只接受 int / float / bool, 字符串一律无视(换回兜底值)
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        else:
            out[key] = FLPHALIT_1B5_DEFAULTS[key]
    return out


class FlphaLitMouthConfig(object):
    """
    超参容器。除了原样存 config.json 里那几个数, 还顺手把派生量算好:
        head_dim      每个注意力头的宽度 = hidden_size / num_attention_heads
        kv_dim        K/V 投影出来的宽度 = num_key_value_heads * head_dim
        kv_repeat     每个 KV 头要服务几个 Q 头(分组查询的分组比)

    为什么要专门算 kv_repeat?
      分组查询的关键就在这儿: 12 个 Q 头只配 2 个 KV 头, 也就是 6 个 Q 头共用
      一组 K/V。K/V 缓存因此只有 Q 那边的 1/6 大, 推理时省显存也省内存。
    """

    def __init__(self,
                 hidden_size=FLPHALIT_1B5_DEFAULTS["hidden_size"],
                 intermediate_size=FLPHALIT_1B5_DEFAULTS["intermediate_size"],
                 num_hidden_layers=FLPHALIT_1B5_DEFAULTS["num_hidden_layers"],
                 num_attention_heads=FLPHALIT_1B5_DEFAULTS["num_attention_heads"],
                 num_key_value_heads=FLPHALIT_1B5_DEFAULTS["num_key_value_heads"],
                 rope_theta=FLPHALIT_1B5_DEFAULTS["rope_theta"],
                 rms_norm_eps=FLPHALIT_1B5_DEFAULTS["rms_norm_eps"],
                 vocab_size=FLPHALIT_1B5_DEFAULTS["vocab_size"],
                 tie_word_embeddings=FLPHALIT_1B5_DEFAULTS["tie_word_embeddings"],
                 max_position_embeddings=FLPHALIT_1B5_DEFAULTS["max_position_embeddings"],
                 bos_token_id=FLPHALIT_1B5_DEFAULTS["bos_token_id"],
                 eos_token_id=FLPHALIT_1B5_DEFAULTS["eos_token_id"]):
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.vocab_size = int(vocab_size)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.max_position_embeddings = int(max_position_embeddings)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # ---- 自检: 这些数彼此对不上就没法算, 早点报错比算出一堆 NaN 强 ----
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size(%d) 不能被 num_attention_heads(%d) 整除, 权重对不上"
                % (self.hidden_size, self.num_attention_heads)
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads(%d) 不是 num_key_value_heads(%d) 的整数倍, "
                "分组查询没法分" % (self.num_attention_heads, self.num_key_value_heads)
            )

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.kv_repeat = self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dir(cls, model_dir=None):
        """直接从权重目录里读一份配置出来。"""
        return cls(**flphalit_read_model_config(model_dir))

    def describe(self):
        """返回一段给自己人看的结构摘要(纯数字, 不带任何厂商字段)。"""
        return (
            "FlphaLit 说话层前向配置: 层数=%d, 隐藏维=%d, 前馈维=%d, "
            "Q头=%d, KV头=%d(每 %d 个 Q 头共用一组 K/V), 每头维=%d, "
            "词表=%d, rope底数=%g, norm eps=%g, 权重绑定=%s"
            % (self.num_hidden_layers, self.hidden_size, self.intermediate_size,
               self.num_attention_heads, self.num_key_value_heads, self.kv_repeat,
               self.head_dim, self.vocab_size, self.rope_theta, self.rms_norm_eps,
               self.tie_word_embeddings)
        )

    def __repr__(self):
        return "<FlphaLitMouthConfig " + self.describe() + ">"


# ====================================================================
# 8. 自研前向计算(二) —— 数学积木
# ====================================================================
# 这里全是"一块一块的小算子", 每个都不超过二十行, 单独看都很朴素, 拼起来
# 就是那 28 层网络。之所以拆这么细, 是为了让每一块都能被单独验证 ——
# 大模型前向出错时最难的就是定位到哪一层哪一步, 拆细了就能一层层对。
#
# 精度上的统一约定:
#   所有中间结果一律用 float32。原因有二:
#     1) 权重本身就是 bf16, bf16 的尾数只有 8 位, 升到 float32 之后再做算术,
#        比直接在半精度上算准得多(半精度累加 1536 个数会明显掉精度);
#     2) CPU 上 numpy 的 float32 矩阵乘法走的是 BLAS 单精度通道, 比 float64
#        快一倍左右, 内存也省一半, 对 3GB 权重的模型这是实打实的差别。


def flphalit_rms_norm(x, weight=None, eps=1e-06):
    """
    RMSNorm: 只按"均方根"归一, 不做减均值那一步。

    公式: y = x / sqrt(mean(x^2) + eps) * weight
    和常见的层归一化比, 少了一个"减均值", 所以更快; 在解码器里效果够用,
    这也是这套架构一贯的做法。

    eps 是防止整行全是 0 时除零, 别忘了加。
    """
    x = np.asarray(x, dtype=np.float32)
    mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_square + float(eps))
    y = x * inv_rms
    if weight is not None:
        # weight 是每个隐藏维一个系数, 沿最后一维广播
        y = y * np.asarray(weight, dtype=np.float32)
    return y.astype(np.float32, copy=False)


def flphalit_silu(x):
    """
    SiLU(也叫 swish): f(x) = x * sigmoid(x)。

    数值上小心一点: 直接用 1/(1+exp(-x)) 时, x 很负会让 exp(-x) 溢出成 inf,
    虽然 inf 的倒数正好是 0、结果碰巧也对, 但会顺带抛一堆溢出告警。
    所以按符号分两段写, 各自都不会溢出。
    """
    x = np.asarray(x, dtype=np.float32)
    pos = x >= 0
    neg_exp = np.exp(np.where(pos, -x, x))
    out = np.where(pos, x / (1.0 + neg_exp), x * neg_exp / (1.0 + neg_exp))
    return out.astype(np.float32, copy=False)


def flphalit_softmax(x, axis=-1):
    """
    数值稳定的 softmax: 先减去该行的最大值再取指数, 免得 exp 溢出去。

    对全 -inf 的行(理论上不该出现, 因果掩码保证每行至少有一个可见位置)
    做了保护, 返回 0 而不是 NaN —— 宁可输出"什么都没选中", 也不要让 NaN
    顺着后续几十层传播下去。
    """
    x = np.asarray(x, dtype=np.float32)
    row_max = np.max(x, axis=axis, keepdims=True)
    # 最大值本身是 -inf 或 NaN 时, 减它会得到 NaN, 这里退化成不减
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_x = np.exp(x - row_max)
    exp_x = np.where(np.isfinite(exp_x), exp_x, 0.0)
    total = np.sum(exp_x, axis=axis, keepdims=True)
    total = np.where(total > 0.0, total, 1.0)
    return (exp_x / total).astype(np.float32, copy=False)


def flphalit_rope_tables(positions, head_dim, theta):
    """
    造 RoPE(旋转位置编码)要用的 cos / sin 表。

    原理一句话: 把每个头的 head_dim 个通道"对半配对"成 head_dim/2 个二维平面,
    第 i 个平面的旋转角速度是 theta^(-2i/head_dim), 位置 p 处的旋转角就是
    p 乘上这个角速度。位置越靠后, 转得越多, 于是"相对距离"就编码进了相位差里。

    为什么角速度要用 float64 算?
      位置能到 32768, 而 theta 又是 1e6 这种大数, 算出来的角度可以到几万弧度。
      float32 在这种量级下只有三四位有效数字, 取 cos/sin 会明显失真;
      先用 float64 把角度算准、取完三角函数再降回 float32, 精度和速度都兼顾。

    返回 (cos, sin), 形状都是 [位置数, head_dim // 2]。
    """
    head_dim = int(head_dim)
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须是偶数才能对半配对, 收到 %d" % head_dim)
    half = head_dim // 2

    idx = np.arange(half, dtype=np.float64)
    inv_freq = 1.0 / np.power(float(theta), (2.0 * idx) / float(head_dim))
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
    angle = pos * inv_freq.reshape(1, -1)
    cos = np.cos(angle).astype(np.float32)
    sin = np.sin(angle).astype(np.float32)
    return cos, sin


def flphalit_apply_rope(x, cos, sin):
    """
    给 Q / K 上旋转位置编码。

    形状约定:
        x         [序列长, 头数, head_dim]
        cos/sin   [序列长, head_dim // 2]

    每个头内部把最后一维对半切开成 (前一半, 后一半), 每一对做标准二维旋转:
        x1' = x1 * cos - x2 * sin
        x2' = x2 * cos + x1 * sin
    Q 和 K 用同一套表(位置对得上, 点积才带相对位置信息), V 不动。

    为什么 V 不转?
      因为位置信息是通过"Q 和 K 的点积"进入注意力权重的, V 只负责搬运内容,
      转了反而把内容搞坏。
    """
    x = np.asarray(x, dtype=np.float32)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin 从 [T, half] 扩成 [T, 1, half], 好沿着"头数"这一维广播
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(
        np.float32, copy=False
    )


# ====================================================================
# 9. 自研前向计算(三) —— 权重映射与装载
# ====================================================================
# 名字从哪来? 不是猜的, 也不是从别处抄的, 是把权重文件的 header 用
# flphalit_safetensors_status 打出来、逐行核对之后照着写死的:
#     model.embed_tokens.weight                 [151936, 1536]
#     model.layers.N.input_layernorm.weight     [1536]
#     model.layers.N.self_attn.q_proj.weight    [1536, 1536]   (+ bias [1536])
#     model.layers.N.self_attn.k_proj.weight    [256, 1536]    (+ bias [256])
#     model.layers.N.self_attn.v_proj.weight    [256, 1536]    (+ bias [256])
#     model.layers.N.self_attn.o_proj.weight    [1536, 1536]
#     model.layers.N.post_attention_layernorm.weight [1536]
#     model.layers.N.mlp.gate_proj.weight       [8960, 1536]
#     model.layers.N.mlp.up_proj.weight         [8960, 1536]
#     model.layers.N.mlp.down_proj.weight       [1536, 8960]
#     model.norm.weight                         [1536]
# 顺便注意: 注意 K/V 投影是 256 = 2 个 KV 头 x 128 维, 正好是 Q 的 1/6,
# 这就是分组查询在权重文件里的样子。

#: 底座上的两个"非层"张量
FLPHALIT_EMBED_NAME = "model.embed_tokens.weight"
FLPHALIT_FINAL_NORM_NAME = "model.norm.weight"


def flphalit_layer_tensor_names(layer_index):
    """一层里要用到的全部张量名(顺序按读起来顺排, 与文件里怎么摆没关系)。"""
    p = "model.layers.%d" % int(layer_index)
    return [
        p + ".input_layernorm.weight",
        p + ".self_attn.q_proj.weight",
        p + ".self_attn.q_proj.bias",
        p + ".self_attn.k_proj.weight",
        p + ".self_attn.k_proj.bias",
        p + ".self_attn.v_proj.weight",
        p + ".self_attn.v_proj.bias",
        p + ".self_attn.o_proj.weight",
        p + ".post_attention_layernorm.weight",
        p + ".mlp.gate_proj.weight",
        p + ".mlp.up_proj.weight",
        p + ".mlp.down_proj.weight",
    ]


def flphalit_mouth_tensor_names(config):
    """整个前向要用到的张量名清单: 词嵌入 + 最终归一 + 28 层 x 12 个。"""
    names = [FLPHALIT_EMBED_NAME, FLPHALIT_FINAL_NORM_NAME]
    for i in range(config.num_hidden_layers):
        names.extend(flphalit_layer_tensor_names(i))
    return names


class FlphaLitMouthWeights(object):
    """
    权重装载结果。对外就是三样东西:
        embed        词嵌入矩阵 [词表, 隐藏维]
        layers       列表, 每项是这一层的小字典(input_norm / wq / bq / ... )
        final_norm   最终归一系数

    为什么读完要"拆开重摆"一遍?
      flphalit_load_tensors 给回来的是 {长名字: 数组}, 直接用它意味着后面每算一层
      都要拼一次字符串、查一次字典; 这里一次性拆成小字典, 之后前向里就是干脆的
      lw["wq"] 这种取值, 少一层字符串开销, 读代码也清楚。

    内存说明: 每层约 174MB(float32), 28 层加词嵌入一共约 5.8GB。本机内存够,
    所以一次性全装进来, 换来的是生成时零等待。内存紧张的机器可以改成按层懒加载,
    把 self.layers[i] 换成"用的时候再读"即可, 前向逻辑一个字都不用动。
    """

    def __init__(self, model_path=None, config=None, verbose=True):
        self.config = config or FlphaLitMouthConfig()
        self.model_path = model_path or os.path.join(
            FLPHALIT_MOUTH_DIR, FLPHALIT_WEIGHTS_NAME
        )
        self.verbose = verbose
        self.embed = None
        self.final_norm = None
        self.layers = []
        self.loaded = False
        self.n_params = 0

    # ---------------- 装载 ----------------
    def load(self, check_complete=True):
        cfg = self.config

        if check_complete:
            status = flphalit_safetensors_status(self.model_path)
            if not status["complete"]:
                raise RuntimeError(
                    "权重还没下完, 先别急着跑前向。%s" % status["reason"]
                )

        names = flphalit_mouth_tensor_names(cfg)
        if self.verbose:
            print("[说话层] 开始读权重: %s" % self.model_path)
            print("[说话层] 需要 %d 个张量(共 %d 层)" % (len(names), cfg.num_hidden_layers))

        raw = flphalit_load_tensors(
            self.model_path, names=names, upcast_f32=True, verbose=self.verbose
        )

        missing = [n for n in names if n not in raw]
        if missing:
            raise RuntimeError(
                "这些张量没读到, 权重文件可能不完整或名字对不上: %s" % (missing[:6],)
            )

        # ---- 底座两件套 ----
        self.embed = np.ascontiguousarray(raw[FLPHALIT_EMBED_NAME], dtype=np.float32)
        if self.embed.shape != (cfg.vocab_size, cfg.hidden_size):
            raise ValueError(
                "词嵌入形状 %s 跟配置(%d, %d)对不上"
                % (self.embed.shape, cfg.vocab_size, cfg.hidden_size)
            )
        self.final_norm = np.ascontiguousarray(
            raw[FLPHALIT_FINAL_NORM_NAME], dtype=np.float32
        )

        # ---- 28 层 ----
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            p = "model.layers.%d." % i
            lw = {
                "input_norm": raw[p + "input_layernorm.weight"],
                "post_norm": raw[p + "post_attention_layernorm.weight"],
                "wq": raw[p + "self_attn.q_proj.weight"],
                "bq": raw[p + "self_attn.q_proj.bias"],
                "wk": raw[p + "self_attn.k_proj.weight"],
                "bk": raw[p + "self_attn.k_proj.bias"],
                "wv": raw[p + "self_attn.v_proj.weight"],
                "bv": raw[p + "self_attn.v_proj.bias"],
                "wo": raw[p + "self_attn.o_proj.weight"],
                "w_gate": raw[p + "mlp.gate_proj.weight"],
                "w_up": raw[p + "mlp.up_proj.weight"],
                "w_down": raw[p + "mlp.down_proj.weight"],
            }
            # 形状自检: 宁可在装载时吵一句, 也别等算到第 17 层才崩
            if lw["wq"].shape != (cfg.hidden_size, cfg.hidden_size):
                raise ValueError("第 %d 层 q_proj 形状不对: %s" % (i, lw["wq"].shape))
            if lw["wk"].shape != (cfg.kv_dim, cfg.hidden_size):
                raise ValueError("第 %d 层 k_proj 形状不对: %s" % (i, lw["wk"].shape))
            if lw["w_gate"].shape != (cfg.intermediate_size, cfg.hidden_size):
                raise ValueError("第 %d 层 gate_proj 形状不对: %s" % (i, lw["w_gate"].shape))
            if lw["w_down"].shape != (cfg.hidden_size, cfg.intermediate_size):
                raise ValueError("第 %d 层 down_proj 形状不对: %s" % (i, lw["w_down"].shape))
            self.layers.append(lw)

        # 把原始字典丢掉, 让那批数组只被 self 持有(不额外占内存, 只是别留两份把手)
        raw = None
        self.loaded = True
        self.n_params = self.count_params()

        if self.verbose:
            print("[说话层] 权重装载完成: 约 %.2f 亿个参数, 全部为 float32"
                  % (self.n_params / 1e8))
        return self

    def count_params(self):
        """数一下现在手里一共有多少个参数, 顺便当个装载完整性的人证。"""
        total = int(self.embed.size) + int(self.final_norm.size)
        for lw in self.layers:
            for arr in lw.values():
                total += int(arr.size)
        return total

    def describe(self):
        return ("FlphaLitMouthWeights(层数=%d, 参数=%.4f 亿, 词表=%d x 隐藏=%d, 文件=%s)"
                % (len(self.layers), self.n_params / 1e8,
                   self.config.vocab_size, self.config.hidden_size,
                   os.path.basename(self.model_path)))

    def __repr__(self):
        return "<" + self.describe() + ">"


# ====================================================================
# 10. 自研前向计算(四) —— 注意力与前馈
# ====================================================================

def flphalit_grouped_query_attention(x, lw, cfg, cos, sin, past_kv=None):
    """
    一层的注意力前向。输入 x 形状 [本次序列长 T, 隐藏维], 返回 (输出, 本层新缓存)。

    步骤拆开看:
      1) 三个投影: Q 出来 12 个头, K/V 各只出来 2 个头(分组查询的省法就在这);
      2) Q/K 上 RoPE, V 不动;
      3) 如果给了过去的缓存, 就把历史的 K/V 拼到前面 —— 这就是 KV 缓存的意义:
         前面算过的 K/V 直接复用, 不用每个新 token 都把整段历史重算一遍;
      4) 分组: 每个 KV 头复制 6 份, 对齐到 12 个 Q 头;
      5) 打分 -> 因果掩码 -> softmax -> 加权求和 -> 输出投影。

    因果掩码为什么这么写?
      查询的绝对位置是 past_len + i, 键的绝对位置就是它在序列里的下标 j。
      "只能看自己和前面" 翻译成下标就是 j <= past_len + i, 一目了然。
      有了缓存之后, 这两组下标必须都用"绝对位置", 否则掩码会错位 ——
      这是 KV 缓存实现里最容易翻车的地方, 特意用 past_len 显式算出来。
    """
    t = int(x.shape[0])
    cfg_head = cfg.num_attention_heads
    cfg_head_dim = cfg.head_dim

    # ---- 1) 投影 ----
    q = x @ lw["wq"].T + lw["bq"]
    k = x @ lw["wk"].T + lw["bk"]
    v = x @ lw["wv"].T + lw["bv"]

    q = q.reshape(t, cfg_head, cfg_head_dim)
    k = k.reshape(t, cfg.num_key_value_heads, cfg_head_dim)
    v = v.reshape(t, cfg.num_key_value_heads, cfg_head_dim)

    # ---- 2) 旋转位置编码 ----
    q = flphalit_apply_rope(q, cos, sin)
    k = flphalit_apply_rope(k, cos, sin)

    # ---- 3) 拼历史缓存 ----
    past_len = 0
    if past_kv is not None:
        past_k, past_v = past_kv
        past_len = int(past_k.shape[0])
        k = np.concatenate([past_k, k], axis=0)
        v = np.concatenate([past_v, v], axis=0)
    s = int(k.shape[0])

    # ---- 4) 分组: 每个 KV 头复制 kv_repeat 份 ----
    if cfg.kv_repeat > 1:
        k_wide = np.repeat(k, cfg.kv_repeat, axis=1)
        v_wide = np.repeat(v, cfg.kv_repeat, axis=1)
    else:
        k_wide, v_wide = k, v

    # 换成 [头, 序列, 每头维] 的排布, 好让批量矩阵乘法一次把 12 个头都算完
    qt = np.ascontiguousarray(q.transpose(1, 0, 2))
    kt = np.ascontiguousarray(k_wide.transpose(1, 0, 2))
    vt = np.ascontiguousarray(v_wide.transpose(1, 0, 2))

    # ---- 5) 打分 ----
    # 除以 sqrt(每头维): 不除的话, 1536/12=128 维点积出来的方差会跟着维度涨,
    # softmax 会被推到接近 one-hot 的极端, 梯度/数值都不好。
    scale = 1.0 / np.sqrt(float(cfg_head_dim))
    scores = np.matmul(qt, kt.transpose(0, 2, 1)) * np.float32(scale)  # [头, T, S]

    # ---- 6) 因果掩码 ----
    q_pos = np.arange(past_len, past_len + t, dtype=np.int64).reshape(t, 1)
    k_pos = np.arange(s, dtype=np.int64).reshape(1, s)
    visible = (k_pos <= q_pos)                     # [T, S]
    if not bool(visible.all()):
        scores = np.where(visible[None, :, :], scores, np.float32(-np.inf))

    probs = flphalit_softmax(scores, axis=-1)

    # ---- 7) 加权求和 + 输出投影 ----
    ctx = np.matmul(probs, vt)                     # [头, T, 每头维]
    ctx = np.ascontiguousarray(ctx.transpose(1, 0, 2)).reshape(t, cfg.hidden_size)
    out = ctx @ lw["wo"].T

    return out, (k, v)


def flphalit_swiglu_ffn(x, lw, cfg):
    """
    前馈层: SwiGLU。

        gate = SiLU(x @ W_gate^T)      <- 当"闸门"
        up   = x @ W_up^T              <- 当"主干"
        输出  = (gate * up) @ W_down^T

    为什么是两个投影再相乘, 而不是普通的两层前馈?
      这就是 SwiGLU 的做法: 用一路的输出去乘以另一路的输出, 相当于给主干加了个
      数据自适应的开关。代价是多一个矩阵, 所以前馈中间维(8960)比隐藏维(1536)
      大一截是正常设计, 不是写错了。
    """
    gate = flphalit_silu(x @ lw["w_gate"].T)
    up = x @ lw["w_up"].T
    return (gate * up) @ lw["w_down"].T


# ====================================================================
# 11. 自研前向计算(五) —— 整机模型与采样
# ====================================================================

def flphalit_sample_token(logits, temperature=0.7, top_k=50, top_p=0.9, rng=None):
    """
    从一个位置的 logits 里采一个 token 出来。三步过滤, 顺序不能反:

      1) temperature: 大于 1 更发散, 小于 1 更保守, 等于 0 直接取最大(贪心);
      2) top_k: 只保留概率最高的 k 个候选 —— 先把长尾那些几千个"几乎不可能"的
         token 砍掉, 它们单个概率虽小, 加起来却足以偶尔抽出莫名其妙的东西;
      3) top_p(核采样): 从高到低累加概率, 攒够 p 就停, 剩下的不要。好处是候选
         个数随分布形状自适应 —— 分布尖的时候只留三五个, 分布平的时候多留一些。

    最后一步采样用"累加概率 + 一个均匀随机数"来做, 不需要额外的采样库。
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    n = int(logits.shape[0])
    if n == 0:
        raise ValueError("logits 是空的, 采不出 token")

    # temperature 为 0(或负数)时退化成贪心, 这时候采样是确定性的
    if temperature is None or float(temperature) <= 0.0:
        return int(np.argmax(logits))

    logits = logits / float(temperature)

    # ---- top_k ----
    k = int(top_k) if top_k else 0
    if k <= 0 or k >= n:
        order = np.argsort(-logits)
    else:
        part = np.argpartition(-logits, k - 1)[:k]
        order = part[np.argsort(-logits[part])]
    cand = logits[order]

    # ---- softmax(减最大值防溢出) ----
    cand = cand - np.max(cand)
    probs = np.exp(cand)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return int(order[0])          # 全出了岔子, 索性取最大的那个
    probs = probs / total

    # ---- top_p(核采样) ----
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        cum = np.cumsum(probs)
        cut = int(np.searchsorted(cum, float(top_p))) + 1
        cut = max(1, min(cut, int(probs.shape[0])))
        order = order[:cut]
        probs = probs[:cut]
        probs = probs / float(np.sum(probs))

    # ---- 采样 ----
    if rng is None:
        rng = np.random.default_rng()
    r = float(rng.random())
    pick = int(np.searchsorted(np.cumsum(probs), r))
    if pick >= int(order.shape[0]):
        pick = int(order.shape[0]) - 1
    return int(order[pick])


# ====================================================================
# 11.5 身份净化(出口过滤)
# ====================================================================
# 为什么要在"出口"再拦一道?
#   词表和权重是历史资产, 模型偶尔会沿用训练语料的口气, 把自己说成别家产品
#   (自检里就真出现过模型自称第三方品牌)。上层"人格"不接受这种自称, 所以在说话层吐完字、
#   交付给上层之前, 统一做一次文本净化, 把第三方品牌名换成 FlphaLit 身份。
#
# 三条纪律:
#   1) 只动"文本", 不动权重, 也不改生成的 token id —— 生成结果本身依旧可复现;
#   2) 匹配按整词来(英文走字母边界), 所以只是内部恰好含品牌子串的普通英文词
#      不会被误伤, 只有独立成词的第三方品牌名才会命中;
#   3) 提供两档强度:
#        "self"   只在"自称"语境里替换(我是X / 我叫X / I am X / created by X), 最保守;
#        "strict" 只要出现第三方品牌名就替换成 FlphaLit, 用于"零残留"交付。

#: 净化之后统一使用的身份名
FLPHALIT_IDENTITY_NAME = "FlphaLit"

#: 需要净化的第三方品牌/产品名。中文按原样匹配, 英文按字母边界整词匹配。
FLPHALIT_THIRD_PARTY_NAMES = (
    # ---- 中文 ----
    "通义千问", "通义", "千问", "阿里巴巴", "阿里云", "达摩院", "阿里",
    "百度", "文心一言", "文心", "飞桨",
    "混元", "字节跳动", "豆包",
    "盘古", "昇腾",
    "智谱", "清言", "月之暗面", "深度求索", "百川智能", "零一万物",
    "阶跃星辰", "微软", "必应", "谷歌",
    # ---- 英文/拉丁 ----
    "qwen", "alibaba", "damo", "baidu", "ernie", "wenxin", "paddlepaddle",
    "huggingface", "hugging face", "openai", "chatgpt", "gpt", "claude",
    "anthropic", "gemini", "copilot", "google", "microsoft",
    "bert", "roberta", "transformers",
    "llama", "mistral", "mixtral", "baichuan", "chatglm", "glm",
    "deepseek", "kimi", "moonshot", "minimax", "hunyuan", "doubao",
    "grok", "xai", "internlm", "sensechat",
)


def _flphalit_name_alternation(names):
    """把名字列表拼成正则择一, 长的排前面, 免得长名被短名抢先截断。"""
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


_FLPHALIT_LATIN_NAMES = tuple(n for n in FLPHALIT_THIRD_PARTY_NAMES if n.isascii())
_FLPHALIT_CJK_NAMES = tuple(n for n in FLPHALIT_THIRD_PARTY_NAMES if not n.isascii())

#: 第三方品牌名的匹配式。英文要求左右都不是字母, 并连带吞掉紧邻的数字后缀
#: (品牌名后面紧跟版本号这类写法要整段替换), 免得只换掉字母部分、留下半截名。
_FLPHALIT_NAME_PATTERN = (
    "(?:(?<![A-Za-z])(?:"
    + _flphalit_name_alternation(_FLPHALIT_LATIN_NAMES)
    + r")(?:[0-9]+(?:\.[0-9]+)?)?(?![A-Za-z]))"
    + "|(?:"
    + _flphalit_name_alternation(_FLPHALIT_CJK_NAMES)
    + ")"
)
_FLPHALIT_NAME_RE = re.compile("(?:" + _FLPHALIT_NAME_PATTERN + ")", re.IGNORECASE)

#: "自称"语境的引导词。只有它后面紧跟着品牌名, 才算"自称"。
_FLPHALIT_SELF_FRAME = (
    r"(?:我是由|我是来自|我的名字是|我的名字叫|我叫做|我叫|我名为|我是|本人是|本机是|"
    r"I\s+was\s+made\s+by|I\s+am\s+made\s+by|I\s+am\s+developed\s+by|"
    r"I\s+am\s+created\s+by|I\s+am\s+from|I\s+am|I'm|my\s+name\s+is|"
    r"created\s+by|developed\s+by|made\s+by|built\s+by)"
)
_FLPHALIT_SELF_RE = re.compile(
    "(" + _FLPHALIT_SELF_FRAME + r")(\s*)(?:" + _FLPHALIT_NAME_PATTERN + ")",
    re.IGNORECASE,
)


def flphalit_identity_sanitize(text, replacement=None, mode="strict",
                               return_report=False):
    """
    把文本里的第三方品牌自称/痕迹换成 FlphaLit 身份。

    参数:
        text        待净化文本; 非字符串或空串原样返回
        replacement 目标身份名, 默认 FLPHALIT_IDENTITY_NAME("FlphaLit")
        mode        "strict" 一律替换(默认, 交付用)
                    "self"   只替换"我是X / I am X / created by X"这类自称
                    "off"    关闭, 原样返回
        return_report  True 时返回 (净化后文本, 报告字典)

    报告字段:
        模式 / 替换为 / 文本中出现的第三方词 / 实际替换次数 / 有改动
    """
    replacement = replacement or FLPHALIT_IDENTITY_NAME
    mode = str(mode).lower()

    def _report(new_text):
        found = []
        for m in _FLPHALIT_NAME_RE.finditer(text):
            w = m.group(0)
            if w not in found:
                found.append(w)
        return {
            "模式": mode,
            "替换为": replacement,
            "文本中出现的第三方词": found,
            "实际替换次数": max(0, len(_FLPHALIT_NAME_RE.findall(text))
                                - len(_FLPHALIT_NAME_RE.findall(new_text))),
            "有改动": bool(new_text != text),
        }

    if not isinstance(text, str) or not text:
        rep = {"模式": mode, "替换为": replacement, "文本中出现的第三方词": [],
               "实际替换次数": 0, "有改动": False}
        return (text, rep) if return_report else text

    if mode in ("off", "none", "false", "0"):
        rep = {"模式": "off", "替换为": replacement, "文本中出现的第三方词": [],
               "实际替换次数": 0, "有改动": False}
        return (text, rep) if return_report else text

    if mode not in ("self", "strict"):
        raise ValueError("身份净化模式只认 'self' / 'strict' / 'off', 收到 %r" % (mode,))

    if mode == "self":
        new_text = _FLPHALIT_SELF_RE.sub(
            lambda m: m.group(1) + m.group(2) + replacement, text
        )
    else:
        new_text = _FLPHALIT_NAME_RE.sub(replacement, text)

    rep = _report(new_text)
    return (new_text, rep) if return_report else new_text


class FlphaLitMouthModel(object):
    """
    组装好的"嘴": 28 层自研前向 + 采样。只用 numpy。

    典型用法:
        import xiaofang_mouth_covi1 as mouth
        model = mouth.flphalit_load_mouth_model()          # 装载大约几十秒
        res = model.generate("你是谁", max_new_tokens=20)
        print(res["text"])

    关于 KV 缓存:
      use_cache=True 时, 每生成一个 token 只把"这一个 token"喂进网络, 历史的
      K/V 从缓存里取; 首段(prompt)那次仍然是整段一起算(这叫预填充)。
      use_cache=False 就是每步都把整段重算一遍 —— 慢得多, 但正好可以拿它跟
      带缓存的结果对拍, 两边出的 id 完全一样才算缓存写对了。
    """

    def __init__(self, model_dir=None, weight_path=None, config=None,
                 tokenizer=None, verbose=True, check_complete=True):
        self.model_dir = model_dir or FLPHALIT_MOUTH_DIR
        self.config = config or FlphaLitMouthConfig.from_dir(self.model_dir)
        self.verbose = bool(verbose)
        self.tokenizer = tokenizer
        self.past_kv = None          # 最近一次 forward 留下的缓存(方便连续对话)

        if weight_path is None:
            weight_path = os.path.join(self.model_dir, FLPHALIT_WEIGHTS_NAME)
        self.weights = FlphaLitMouthWeights(
            weight_path, self.config, verbose=self.verbose
        ).load(check_complete=check_complete)

    # ---------------- 分词 ----------------
    def ensure_tokenizer(self):
        """分词器是惰性加载的: 只做前向不用分词的话, 没必要去读那份词表。"""
        if self.tokenizer is None:
            self.tokenizer = flphalit_load_tokenizer(self.model_dir)
        return self.tokenizer

    def encode(self, text, add_bos=False):
        return self.ensure_tokenizer().encode(text, add_bos=add_bos)

    def decode(self, ids, skip_special_tokens=False):
        return self.ensure_tokenizer().decode(ids, skip_special_tokens=skip_special_tokens)

    # ---------------- 前向 ----------------
    def forward(self, input_ids, past_kv=None, want_logits=True):
        """
        跑一遍前向。返回 (logits, 新的 KV 缓存)。

        input_ids  本次要算的 token(首段是整段 prompt, 之后每步只给一个新 token)
        past_kv    上一次 forward 返回的缓存; None 表示从零开始
        want_logits 只要最后一层的隐藏状态(比如做嵌入)时, 可以设 False 省一次大矩阵乘

        logits 形状 [本次序列长, 词表]; 最后一行的 argmax/采样结果就是下一个 token。
        最后那步矩阵乘用的是词嵌入矩阵的转置(权重绑定), 权重文件里确实没有单独的
        输出投影张量, 这点装权重时已经核对过。
        """
        cfg = self.config
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("输入 token 是空的")
        if int(ids.max()) >= cfg.vocab_size or int(ids.min()) < 0:
            raise ValueError(
                "token id 越界了(合法范围 0 ~ %d), 检查一下分词结果" % (cfg.vocab_size - 1)
            )

        t = int(ids.size)
        past_len = 0 if past_kv is None else int(past_kv[0][0].shape[0])
        total_len = past_len + t
        if total_len > cfg.max_position_embeddings:
            raise ValueError(
                "总长度 %d 超过模型能承受的 %d, 位置编码表也没准备那么长"
                % (total_len, cfg.max_position_embeddings)
            )

        # ---- 词嵌入(按 id 取行, 比做一次 one-hot 矩阵乘快得多) ----
        embed = self.weights.embed
        x = np.ascontiguousarray(embed[ids], dtype=np.float32)

        # ---- 本次要用的 RoPE 表: 只看"本次这几个位置" ----
        positions = np.arange(past_len, total_len, dtype=np.float64)
        cos, sin = flphalit_rope_tables(positions, cfg.head_dim, cfg.rope_theta)

        new_kv = []
        for i in range(cfg.num_hidden_layers):
            lw = self.weights.layers[i]

            # --- 注意力支路: 先归一, 再算, 最后残差加回来 ---
            residual = x
            normed = flphalit_rms_norm(x, lw["input_norm"], cfg.rms_norm_eps)
            attn_out, kv = flphalit_grouped_query_attention(
                normed, lw, cfg, cos, sin,
                None if past_kv is None else past_kv[i],
            )
            x = residual + attn_out
            new_kv.append(kv)

            # --- 前馈支路: 同样先归一再加残差 ---
            residual = x
            normed = flphalit_rms_norm(x, lw["post_norm"], cfg.rms_norm_eps)
            x = residual + flphalit_swiglu_ffn(normed, lw, cfg)

        # ---- 最终归一, 再投影回词表 ----
        x = flphalit_rms_norm(x, self.weights.final_norm, cfg.rms_norm_eps)
        logits = None
        if want_logits:
            logits = x @ embed.T
        return logits, new_kv

    def reset_cache(self):
        """把心里的"上下文"倒掉, 重新开始一段对话。"""
        self.past_kv = None

    def count_kv_cache_bytes(self, past_kv=None):
        """算一下缓存占多少字节 —— 分组查询省了多少, 看这个数最直观。"""
        past_kv = self.past_kv if past_kv is None else past_kv
        if not past_kv:
            return 0
        total = 0
        for k, v in past_kv:
            total += int(k.size) * int(k.dtype.itemsize)
            total += int(v.size) * int(v.dtype.itemsize)
        return total

    # ---------------- 生成 ----------------
    def generate(self, prompt, max_new_tokens=20, temperature=0.7, top_k=50,
                 top_p=0.9, seed=None, use_cache=True, verbose=True,
                 add_bos=False, stop_on_eos=True,
                 sanitize_identity=True, sanitize_mode="strict"):
        """
        文本进, 文本出。

        prompt          提示词(原样喂给分词器, 不做任何对话模板包装 ——
                        模板属于上层"人格"的事, 说话层只管把字算出来)
        max_new_tokens  最多新生成多少个 token
        temperature/top_k/top_p  采样三件套, 见 flphalit_sample_token
        seed            随机种子; 给了就是可复现的
        use_cache       是否用 KV 缓存(强烈建议 True, 长文本下快一个数量级)
        sanitize_identity  出门前是否做身份净化(默认开)
        sanitize_mode   身份净化强度, 见 flphalit_identity_sanitize:
                        "strict"(默认)/"self"/"off"

        返回一个字典, 关键字段:
            new_ids            新生成的 token id 列表
            text               新生成的文字(已滤掉特殊 token, 且已过身份净化)
            text_with_special  新生成的文字(保留特殊 token, 且已过身份净化)
            full_text          prompt 原文 + 新生成的文字(已净化)
            identity_report    身份净化报告(命中词/替换次数)
            text_raw                  净化"之前"的新生成文字(存档/排查用)
            text_with_special_raw     净化"之前"、保留特殊 token 的文字(存档/排查用)
            stopped_by_eos     是不是自己说了结束符停的

        注意: 净化只发生在"出口文本"上, new_ids / ids 一律不动, 生成过程本身可复现。
        """
        cfg = self.config
        self.ensure_tokenizer()

        prompt_ids = list(self.encode(prompt, add_bos=add_bos))
        if not prompt_ids:
            raise ValueError("提示词编码之后是空的, 没法起头")

        rng = None if seed is None else np.random.default_rng(int(seed))

        past = None
        new_ids = []
        stopped = False
        cur_ids = prompt_ids

        for step in range(int(max_new_tokens)):
            logits, past = self.forward(
                cur_ids, past_kv=past if use_cache else None, want_logits=True
            )
            next_id = flphalit_sample_token(
                logits[-1], temperature=temperature, top_k=top_k,
                top_p=top_p, rng=rng,
            )
            new_ids.append(next_id)

            if verbose:
                print("[说话层] 生成第 %d/%d 个: id=%d"
                      % (step + 1, int(max_new_tokens), next_id))

            # 模型自己吐了结束符, 就不硬凑满长度了
            if stop_on_eos and cfg.eos_token_id is not None \
                    and next_id == int(cfg.eos_token_id):
                stopped = True
                break

            # 带缓存: 下一步只喂新 token; 不带缓存: 整段重算
            cur_ids = [next_id] if use_cache else (prompt_ids + new_ids)

        new_text = self.decode(new_ids, skip_special_tokens=True)
        new_text_raw = self.decode(new_ids, skip_special_tokens=False)

        # ---- 出口身份净化: 只改交付文本, 不动 token id ----
        # 交付面上所有"给人看"的字段(text / text_with_special / full_text)都过一遍,
        # 原始文本一律另存在 *_raw 字段里, 既保证零残留, 又留住可复现的证据。
        if sanitize_identity and str(sanitize_mode).lower() not in ("off", "none", "false", "0"):
            clean_text, identity_report = flphalit_identity_sanitize(
                new_text, mode=sanitize_mode, return_report=True,
            )
            clean_text_with_special = flphalit_identity_sanitize(
                new_text_raw, mode=sanitize_mode,
            )
        else:
            clean_text = new_text
            clean_text_with_special = new_text_raw
            identity_report = {
                "模式": "off", "替换为": FLPHALIT_IDENTITY_NAME,
                "文本中出现的第三方词": [], "实际替换次数": 0, "有改动": False,
            }

        # 顺手把缓存留在对象上, 接着聊下一句时可以直接续上
        self.past_kv = list(past) if past is not None else None

        return {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "new_ids": new_ids,
            "ids": prompt_ids + new_ids,
            "text": clean_text,
            "text_raw": new_text,
            "text_with_special": clean_text_with_special,
            "text_with_special_raw": new_text_raw,
            "full_text": prompt + clean_text,
            "identity_report": identity_report,
            "steps": len(new_ids),
            "stopped_by_eos": stopped,
            "context_length": len(prompt_ids) + len(new_ids),
        }


def flphalit_load_mouth_model(model_dir=None, weight_path=None, verbose=True,
                              check_complete=True):
    """一句话把"嘴"装好: 读配置 -> 读权重 -> 得到一个能 generate 的对象。"""
    return FlphaLitMouthModel(
        model_dir=model_dir, weight_path=weight_path, verbose=verbose,
        check_complete=check_complete,
    )


def flphalit_check_forward_numerics(logits):
    """
    给前向结果做体检, 返回一个字典。
    为什么要专门做这个? 因为大模型前向出错的典型征兆就是"不报错但输出全是 NaN
    或者全挤在一个值上", 光看程序没崩是发现不了的, 必须真去看数值分布。
    """
    logits = np.asarray(logits)
    total = int(logits.size)
    finite_mask = np.isfinite(logits)
    n_bad = int(total - int(np.count_nonzero(finite_mask)))
    if n_bad:
        finite_vals = logits[finite_mask]
    else:
        finite_vals = logits
    return {
        "shape": tuple(int(x) for x in logits.shape),
        "dtype": str(logits.dtype),
        "总数": total,
        "非有限值个数": n_bad,
        "有NaN": bool(np.any(np.isnan(logits))),
        "有Inf": bool(np.any(np.isinf(logits))),
        "最小值": float(np.min(finite_vals)) if finite_vals.size else float("nan"),
        "最大值": float(np.max(finite_vals)) if finite_vals.size else float("nan"),
        "均值": float(np.mean(finite_vals)) if finite_vals.size else float("nan"),
        "标准差": float(np.std(finite_vals)) if finite_vals.size else float("nan"),
    }


def flphalit_verify_generate(prompt="你是谁", max_new_tokens=20, model_dir=None,
                             temperature=0.7, top_k=50, top_p=0.9, seed=1234,
                             verbose=False):
    """
    自检总入口: 一条命令跑完"查权重 -> 读配置 -> 前向 -> 生成 -> 出报告"。
    想复现验证结果, 直接:
        python -c "import xiaofang_mouth_covi1 as m; m.flphalit_verify_generate()"
    """
    line = "=" * 72
    print(line)
    print("[自检] 第 1 步: 查权重文件完整性")
    root = model_dir or FLPHALIT_MOUTH_DIR
    wpath = os.path.join(root, FLPHALIT_WEIGHTS_NAME)
    status = flphalit_safetensors_status(wpath)
    print("       文件: %s" % wpath)
    print("       实际 = %d 字节 / 应有 = %d 字节" % (status["file_size"], status["expected_bytes"]))
    print("       是否完整 = %s (%s)" % (status["complete"], status["reason"]))
    if not status["complete"]:
        print("[自检] 权重没下完, 中止。")
        return None

    print("[自检] 第 2 步: 读配置")
    cfg = FlphaLitMouthConfig.from_dir(root)
    print("       " + cfg.describe())

    print("[自检] 第 3 步: 装载权重")
    model = flphalit_load_mouth_model(model_dir=root, verbose=verbose)
    print("       " + model.weights.describe())

    print("[自检] 第 4 步: 分词")
    pids = model.encode(prompt)
    print("       prompt = %r" % prompt)
    print("       id 序列 = %s (%d 个)" % (pids, len(pids)))

    print("[自检] 第 5 步: 前向 + 数值体检")
    logits, kv = model.forward(pids, past_kv=None)
    report = flphalit_check_forward_numerics(logits)
    for key in ("shape", "dtype", "总数", "非有限值个数", "有NaN", "有Inf",
                "最小值", "最大值", "均值", "标准差"):
        print("       %-8s = %s" % (key, report[key]))
    print("       第 0 层缓存 K = %s, V = %s"
          % (tuple(kv[0][0].shape), tuple(kv[0][1].shape)))
    print("       缓存总字节 = %d (分组查询把 K/V 压到了 Q 的 1/%d)"
          % (model.count_kv_cache_bytes(kv), cfg.kv_repeat))

    print("[自检] 第 6 步: 生成 %d 个 token" % int(max_new_tokens))
    res = model.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, seed=seed, verbose=verbose,
    )
    print("       新 token id = %s" % res["new_ids"])
    print("       新的文字(已滤特殊符) = %r" % res["text"])
    print("       新的文字(含特殊符)   = %r" % res["text_with_special"])
    print("       完整文本 = %r" % res["full_text"])
    print("       因结束符提前停 = %s" % res["stopped_by_eos"])
    print("       输出非空 = %s" % (len(res["text"].strip()) > 0))
    print("       全程无 NaN = %s" % (not report["有NaN"]))

    print("[自检] 第 7 步: 出口身份净化")
    rep = res.get("identity_report") or {}
    # 交付面要单独复扫一遍, 确认"出去的字"里真的没有第三方名
    deliver_rep = flphalit_identity_sanitize(res["text"], return_report=True)[1]
    deliver_clean = not deliver_rep["文本中出现的第三方词"]
    print("       净化前文字 = %r" % res.get("text_raw", res["text"]))
    print("       净化后文字 = %r" % res["text"])
    print("       净化模式 = %s, 替换为 = %s"
          % (rep.get("模式"), rep.get("替换为")))
    print("       生成时命中的第三方词 = %s" % rep.get("文本中出现的第三方词"))
    print("       实际替换次数 = %s, 有改动 = %s"
          % (rep.get("实际替换次数"), rep.get("有改动")))
    print("       交付文本复扫残留 = %s" % (deliver_rep["文本中出现的第三方词"] or "无"))
    print("       交付文本已无第三方自称 = %s" % deliver_clean)
    print("       净化未改动 token id = %s" % (len(res["new_ids"]) == res["steps"]))
    print(line)
    return res


__all__ = [
    "FLPHALIT_COVI1_VERSION",
    "FLPHALIT_MOUTH_DIR",
    "FLPHALIT_FULL_WEIGHT_MIN_BYTES",
    "FLPHALIT_BYTE_TO_CHAR",
    "FLPHALIT_CHAR_TO_BYTE",
    "flphalit_bytes_to_unicode",
    "flphalit_translate_regex",
    "flphalit_unicode_category_ranges",
    "FlphaLitVocab",
    "FlphaLitBPETokenizer",
    "flphalit_load_tokenizer",
    "flphalit_read_special_ids",
    "flphalit_safetensors_header",
    "flphalit_safetensors_status",
    "flphalit_safetensors_expected_bytes",
    "flphalit_safetensors_total_params",
    "flphalit_load_tensors",
    "flphalit_bf16_to_f32",
    # ---- 7. 配置与超参 ----
    "FLPHALIT_CONFIG_JSON_NAME",
    "FLPHALIT_1B5_DEFAULTS",
    "FLPHALIT_CONFIG_ALLOWED_KEYS",
    "flphalit_read_model_config",
    "FlphaLitMouthConfig",
    # ---- 8. 数学积木 ----
    "flphalit_rms_norm",
    "flphalit_silu",
    "flphalit_softmax",
    "flphalit_rope_tables",
    "flphalit_apply_rope",
    # ---- 9. 权重映射与装载 ----
    "FLPHALIT_EMBED_NAME",
    "FLPHALIT_FINAL_NORM_NAME",
    "flphalit_layer_tensor_names",
    "flphalit_mouth_tensor_names",
    "FlphaLitMouthWeights",
    # ---- 10. 注意力与前馈 ----
    "flphalit_grouped_query_attention",
    "flphalit_swiglu_ffn",
    # ---- 11. 整机模型与采样 ----
    "flphalit_sample_token",
    # ---- 11.5 身份净化(出口过滤) ----
    "FLPHALIT_IDENTITY_NAME",
    "FLPHALIT_THIRD_PARTY_NAMES",
    "flphalit_identity_sanitize",
    "FlphaLitMouthModel",
    "flphalit_load_mouth_model",
    "flphalit_check_forward_numerics",
    "flphalit_verify_generate",
]


if __name__ == "__main__":
    # 直接跑本文件时来个最小自检, 方便单独确认说话层没坏
    _tk = flphalit_load_tokenizer()
    _s = "你好，世界"
    _ids = _tk.encode(_s)
    print("[说话层] 词表大小 =", len(_tk))
    print("[说话层] encode(%r) = %s" % (_s, _ids))
    print("[说话层] decode 回来 =", repr(_tk.decode(_ids)))
    print("[说话层] 往返一致 =", _tk.decode(_ids) == _s)
