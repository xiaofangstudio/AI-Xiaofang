# -*- coding: utf-8 -*-
"""小方 · 旧权重缓存一键清理

为什么会有这个脚本:
    `权重缓存/fw_v1_*.bin` 是"冻结权重"的磁盘缓存 —— 有它启动只要 ~1.5 秒,
    没它就得现场冷编译(1~2 分钟)。它单个约 2.3GB, 而**文件名里带词表大小(V)**,
    词表 = 常用词 ∪ 知识库标题/别名 ∪ 跨会话自学记忆 →
    只要你加过一条知识、或聊出过新生词, 词表就变、签名就变, 于是另开一个新文件,
    旧的再也读不回来, 却一直躺在盘上。实测本机一次就攒了 14 个, 共 32.7GB。

这个脚本做什么(只删废的, 绝不误伤):
    ① 删掉所有 `.bin.tmp` —— 冷编译被 Ctrl+C / 断电打断留下的半截文件, 每个也是 2.3GB 级。
    ② 同档位(签名 d/L/H/F/tie/s/h 全同)只留**最新**的一个 `.bin`, 其余旧词表的删掉。
    ③ 其它档位(Lite / Pro / Ultra)、`title_cache.json` 一律不动。

怎么用:
    双击同目录下的 `清理旧缓存.bat`, 或直接 `python clean_cache.py`。
    主程序本身也会在每次启动时自动做同样的清理, 这个脚本是给"手动一键瘦身"用的。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "权重缓存")


def _human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.2f %s" % (n, u)
        n /= 1024.0
    return "%.2f GB" % n


def main():
    global CACHE_DIR
    # 默认清项目自带的「权重缓存」; 也可以 `python clean_cache.py <某个目录>` 指定
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        CACHE_DIR = os.path.abspath(sys.argv[1])
    print("=" * 56)
    print(" 小方 · 旧权重缓存清理")
    print("=" * 56)
    if not os.path.isdir(CACHE_DIR):
        print("没找到「权重缓存」目录, 无需清理。")
        return 0

    names = os.listdir(CACHE_DIR)
    removed, freed, skipped = [], 0, []

    def _rm(fn, why):
        nonlocal freed
        p = os.path.join(CACHE_DIR, fn)
        try:
            sz = os.path.getsize(p)
            os.remove(p)
            freed += sz
            removed.append((fn, sz, why))
        except Exception as e:
            skipped.append((fn, str(e)))

    # ① 半截文件: 没写完的缓存, 天然是垃圾
    for fn in names:
        if fn.endswith(".bin.tmp"):
            _rm(fn, "半截文件(编译被打断)")

    # ② 同档位旧词表: 分组后每组只留 mtime 最新的那个
    pat = re.compile(r"^fw_v(\d+)_(.*)_V(\d+)_s(\d+)_h(\d+)\.bin$")
    groups = {}
    for fn in names:
        if fn.endswith(".bin.tmp"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(4), m.group(5))
        p = os.path.join(CACHE_DIR, fn)
        try:
            mt = os.path.getmtime(p)
        except Exception:
            mt = 0.0
        groups.setdefault(key, []).append((mt, int(m.group(3)), fn))

    keep_list = []
    for _key, items in sorted(groups.items()):
        items.sort(key=lambda t: (t[0], t[1]), reverse=True)
        keep = items[0]
        keep_list.append(keep)
        for _mt, v, fn in items[1:]:
            _rm(fn, "旧词表 V%d(现用 V%d)" % (v, keep[1]))

    # ---- 报告 ----
    for _mt, v, fn in sorted(keep_list, key=lambda t: t[1], reverse=True):
        p = os.path.join(CACHE_DIR, fn)
        try:
            sz = _human(os.path.getsize(p))
        except Exception:
            sz = "?"
        print("保留  %s  (V%d, %s)" % (fn, v, sz))

    if removed:
        print("-" * 56)
        for fn, sz, why in removed:
            print("删除  %s  (%s, %s)" % (fn, _human(sz), why))
        print("-" * 56)
        print("共清理 %d 个文件, 回收 %s" % (len(removed), _human(freed)))
    else:
        print("-" * 56)
        print("没有需要清理的旧缓存, 目录已经很干净。")

    if skipped:
        print("-" * 56)
        for fn, why in skipped:
            print("跳过  %s  (%s)" % (fn, why))

    other = [f for f in os.listdir(CACHE_DIR)
             if not f.endswith(".bin") and not f.endswith(".bin.tmp")]
    if other:
        print("其它保留: %s" % ", ".join(sorted(other)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("清理出错: %r" % (e,))
        sys.exit(1)
