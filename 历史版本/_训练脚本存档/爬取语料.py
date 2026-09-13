# -*- coding: utf-8 -*-
"""
爬取语料.py —— 自动爬中文网页, 只留"优质句子", TXT 满 5MB 滚动, 全程断点续跑.

v1.1 修的两个坑:
    ① 队列见底就 break —— 爬了 11104 页 / 22.32 MB 就"不动了", 离 50 MB 差一半。
       现在队列一空先"补种"(种子池 + 分页列表页模板), 补满 6 轮还补不出来才收工。
    ② 深链筛得太狠 —— 只有"长得像正文页"的链接才跟, 栏目页/分页列表全被挡了。
       现在深度 3 以内同域链接一律收(是不是正文交给 page_pick 判); 深度上限 4。

产出:  <桌面>\\小方小说\\网络语料\\语料_001.txt, 语料_002.txt ...
       (学习小说.py 的 list_txt 会往下钻一层, 自动把它们收进去)
状态:  <产出目录>\\_crawl\\{state.json, seen.txt, crawl.log}
       (下划线开头, 学习小说.py 的 _skip_name 会跳过, 绝不会被当语料读进去)

优质 / 劣质判别 全部由本脚本内的打分函数完成, 不需要人工挑句子.

用法:
    python 爬取语料.py                      # 默认爬满 50MB 收工
    python 爬取语料.py --target-mb 30 --minutes 60
    python 爬取语料.py --workers 12
    python 爬取语料.py --reset-state        # 清空断点从头来
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

try:
    import urllib3

    urllib3.disable_warnings()
except Exception:
    pass


# ============================================================================
# 0. 路径与常量
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.dirname(HERE)

# 产出目录: 桌面\小方小说\网络语料\
OUT_DIR = os.path.join(DESKTOP, "小方小说", "网络语料")
STATE_DIRNAME = "_crawl"          # 下划线开头 -> 被学习小说.py 跳过

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
    "Connection": "keep-alive",
}

# ---- 种子站点: 只用探针实测"通"的源 ------------------------------------
# v1.1: 种子扩成"站点 × 栏目" —— 一个站点卡住(达到单源页数上限)时, 别的栏目还能供血,
#       不至于整站哑火; 队列见底时 PAGERS 里的分页模板会再灌一批进来。
SEEDS = [
    # 人民网系(同域, 频道各自计数)
    ("人民网-观点",   "http://opinion.people.com.cn/"),
    ("人民网-要闻",   "http://politics.people.com.cn/"),
    ("人民网-社会",   "http://society.people.com.cn/"),
    ("人民网-文化",   "http://culture.people.com.cn/"),
    ("人民网-读书",   "http://book.people.com.cn/"),
    ("人民网-文史",   "http://history.people.com.cn/"),
    # 新华系
    ("新华网",        "http://www.news.cn/"),
    ("新华网-评论",   "http://www.news.cn/comments/"),
    ("新华网-文化",   "http://www.news.cn/culture/"),
    # 光明网
    ("光明网-时评",   "https://guancha.gmw.cn/"),
    ("光明网-文摘",   "https://www.gmw.cn/"),
    ("光明网-文艺",   "https://wenyi.gmw.cn/"),
    # 报刊 / 文学
    ("中国青年网",    "http://www.youth.cn/"),
    ("中青报",        "http://zqb.cyol.com/"),
    ("中国作家网",    "http://www.chinawriter.com.cn/"),
    ("中国新闻网",    "https://www.chinanews.com.cn/"),
    ("澎湃",          "https://www.thepaper.cn/"),
    ("界面新闻",      "https://www.jiemian.com/"),
    ("南方周末",      "http://www.infzm.com/"),
    ("环球网",        "https://www.huanqiu.com/"),
    # 古诗文 / 诗歌 / 字词
    ("古诗文网",      "https://www.gushiwen.cn/"),
    ("古诗文网-名句", "https://www.gushiwen.cn/mingju/"),
    ("中国诗歌网",    "http://www.zgshige.com/"),
    ("句子迷",        "https://www.juzimi.com/"),
    ("汉典",          "https://www.zdic.net/"),
    # 门户 / 书评
    ("豆瓣读书",      "https://book.douban.com/"),
    ("央视网",        "https://www.cctv.com/"),
    ("搜狐",          "https://www.sohu.com/"),
    ("网易",          "https://www.163.com/"),
    ("新浪",          "https://www.sina.com.cn/"),
    ("凤凰网",        "https://www.ifeng.com/"),
    ("百度百科",      "https://baike.baidu.com/"),
    ("知乎",          "https://www.zhihu.com/"),
    ("起点中文网",    "https://www.qidian.com/"),
]

# ---- 队列见底时的"补种": 几个大站的分页列表页模板 (%d = 页码) ----------
# 猜错的页顶多 404 一次, 不花钱; 猜对就能再续几百上千页正文。
PAGERS = [
    ("人民网-观点", "http://opinion.people.com.cn/index%d.html"),
    ("人民网-要闻", "http://politics.people.com.cn/index%d.html"),
    ("人民网-社会", "http://society.people.com.cn/index%d.html"),
    ("人民网-文化", "http://culture.people.com.cn/index%d.html"),
    ("人民网-读书", "http://book.people.com.cn/index%d.html"),
    ("人民网-文史", "http://history.people.com.cn/index%d.html"),
    ("光明网-时评", "https://guancha.gmw.cn/index_%d.htm"),
    ("中国作家网",  "http://www.chinawriter.com.cn/index_%d.html"),
    ("澎湃",        "https://www.thepaper.cn/channel_%d"),
]

MAX_HTML = 2_500_000              # 单页最多读 2.5MB, 防止大页面把内存顶爆
TIMEOUT = (6.8, 12.0)             # (连接, 读取)
STATE_VERSION = 1
FRONTIER_MAX = 200_000            # 队列上限, 防止无限膨胀吃内存
RESEED_ROUNDS = 6                 # 队列见底最多补种几轮, 补不出来才真收工


# ============================================================================
# 1. 优质 / 劣质 判别
# ============================================================================

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
NUMRUN_RE = re.compile(r"\d{3,}")
SPACE_RE = re.compile(r"\s+")

# 命中即判劣质(广告 / 导航 / 版权 / 代码 / 站点设施)
JUNK_WORDS = (
    "版权所有", "京icp", "沪icp", "粤icp", "icp备", "增值电信", "网络文化经营",
    "广告", "赞助", "推广", "优惠券", "立即购买", "点击购买", "扫码关注",
    "扫二维码", "下载app", "下载客户端", "客户端下载", "手机版", "触屏版",
    "登录", "注册", "注销", "找回密码", "用户协议", "隐私政策", "免责声明",
    "关于我们", "联系我们", "加入我们", "意见反馈", "举报", "客服电话",
    "版权声明", "转载", "来源：", "责任编辑", "编辑：", "分享到", "微信",
    "微博", "朋友圈", "qq空间", "评论区", "全部评论", "暂无评论", "热门推荐",
    "相关阅读", "猜你喜欢", "上一篇", "下一篇", "返回首页", "网站地图",
    "友情链接", "cookie", "javascript", "function(", "var ", "css", "html",
    "http://", "https://", "www.", ".com", ".cn/", "©", "版权归",
)

# 这些扩展名的链接一律不跟
BAD_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".zip", ".rar", ".7z", ".gz", ".exe", ".msi", ".apk", ".dmg",
    ".mp3", ".mp4", ".avi", ".mkv", ".flv", ".wav",
    ".css", ".js", ".json", ".xml", ".rss",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)

# 这些路径片段是站内功能页, 一律不跟(登录/搜索/下载/关于我们...)
JUNK_PATH = (
    "/login", "/register", "/signin", "/signup", "/passport", "/user/",
    "/search", "/so.", "/s?wd=", "/rss", "/sitemap", "/about", "/contact",
    "/help", "/feedback", "/download", "/app/", "/apps", "/ad_", "/adv/",
    "/privacy", "/terms", "/job", "/career", "/cart", "/pay", "/vip",
    "javascript:", "mailto:", "tel:",
)

# 优质单行的硬门槛
LINE_MIN, LINE_MAX = 8, 220          # 太短是碎片, 太长是段落堆砌
LINE_MIN_HAN = 6                     # 至少 6 个汉字
LINE_MIN_HAN_RATIO = 0.45            # 汉字占可见字符的比例下限
LINE_MAX_REPEAT_RATIO = 0.40         # 单个字符占比上限(挡掉 "。。。", "啊啊啊")

# 整页门槛: 一篇页面上优质行太少 -> 整页丢弃(导航页/列表页)
PAGE_MIN_LINES = 6
PAGE_MIN_HAN = 300


def line_score(s):
    """优质行判别: 通过返回 True. 这是本脚本的"优质/劣质"唯一裁判. """
    if not (LINE_MIN <= len(s) <= LINE_MAX):
        return False

    han = len(HAN_RE.findall(s))
    if han < LINE_MIN_HAN:
        return False
    if han / float(len(s)) < LINE_MIN_HAN_RATIO:
        return False

    low = s.lower()
    for w in JUNK_WORDS:
        if w in low:
            return False

    # 单字符重复率过高(广告号 / 导航分隔)
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    if max(counts.values()) / float(len(s)) > LINE_MAX_REPEAT_RATIO:
        return False

    # 标点连排 (例如 "。。。。" "，，，，")
    run = 1
    prev = ""
    for ch in s:
        if ch == prev and not HAN_RE.match(ch) and not ch.isalnum():
            run += 1
            if run >= 4:
                return False
        else:
            run = 1
        prev = ch

    # 必须有结尾标点或足够长, 挡掉"标签式短语"
    if len(s) < 14 and s[-1] not in "。！？…”，、；：":
        return False

    return True


def page_pick(lines):
    """整页筛选: 只返回优质行; 优质行/优质字不够 -> 整页弃掉. """
    good = []
    han_total = 0
    for s in lines:
        s = SPACE_RE.sub("", s).strip()
        if not s:
            continue
        if s.startswith(("首页", "导航", "栏目", "当前位置")):
            continue
        if line_score(s):
            good.append(s)
            han_total += len(HAN_RE.findall(s))

    if len(good) < PAGE_MIN_LINES or han_total < PAGE_MIN_HAN:
        return []
    return good


# ============================================================================
# 2. 断点: 原子状态 + 已爬 URL 去重
# ============================================================================

def fp(url):
    return hashlib.blake2b(url.encode("utf-8"), digest_size=8).hexdigest()


def load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("version") == STATE_VERSION:
            return d
    except Exception:
        pass
    return {"version": STATE_VERSION, "frontier": [], "file_index": 1,
            "pages": 0, "written": 0, "src_pages": {}}


def save_state(path, state):
    """先写 .tmp 再 os.replace —— 断电也不会写出半个 json."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Seen(object):
    """已爬 URL 去重: os 层面持久化(seen.txt) + 进程内 inflight 集合."""

    def __init__(self, path):
        self.path = path
        self.done = set()          # 上次运行已经爬过的
        self.inflight = set()      # 本次已排队/已完成的
        self.lock = threading.Lock()
        self.fh = None
        self.pending = 0
        self.disk_ok = True        # v1.2: 落盘一旦坏了就关掉, 不再反复重试
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    k = line.strip()
                    if k:
                        self.done.add(k)

    def has(self, url):
        k = fp(url)
        with self.lock:
            return (k in self.done) or (k in self.inflight)

    def claim(self, url):
        """返回 True 表示"归我爬", 同一 URL 在本次运行中只会被认领一次."""
        k = fp(url)
        with self.lock:
            if k in self.done or k in self.inflight:
                return False
            self.inflight.add(k)
            return True

    def finish(self, url):
        k = fp(url)
        with self.lock:
            self.done.add(k)       # 先记内存 —— 落盘失败也不影响本次去重
            if not self.disk_ok:
                return
            try:
                if self.fh is None:
                    self.fh = open(self.path, "a", encoding="utf-8")
                self.fh.write(k + "\n")
                self.pending += 1
                if self.pending >= 200:
                    self.fh.flush()
                    os.fsync(self.fh.fileno())
                    self.pending = 0
            except Exception as e:
                # v1.2: 受限环境里老 seen.txt 可能续写不了(Errno 9)。这不该拖垮爬虫 ——
                #   内存里的 done 才是去重真身, 落盘只是给下次启动省事。
                log("!! seen.txt 落盘不可用(%s), 本次仅在内存去重" % e)
                self.disk_ok = False
                self.fh = None
                self.pending = 0

    def flush(self):
        with self.lock:
            if self.fh is not None:
                try:
                    self.fh.flush()
                    os.fsync(self.fh.fileno())
                except Exception as e:
                    log("!! seen.txt 落盘不可用(%s), 本次仅在内存去重" % e)
                    self.disk_ok = False
                    self.fh = None


# ============================================================================
# 3. 滚动写盘: 每满 FILE_MB 新建一个 TXT
# ============================================================================

class Writer(object):
    def __init__(self, out_dir, file_mb, target_mb, start_index, base_bytes):
        self.out_dir = out_dir
        self.limit = int(file_mb * 1024 * 1024)
        self.target = int(target_mb * 1024 * 1024)
        self.lock = threading.Lock()
        self.dedupe = set()          # 行级去重(blake2b 8 字节)
        self.idx = int(start_index)
        self.total = int(base_bytes)
        self.lines = 0
        self.fh = None
        self.cur = 0
        self.path = ""
        self._open()

    def _open(self):
        self.path = os.path.join(self.out_dir, "语料_%03d.txt" % self.idx)
        self.cur = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        self.fh = open(self.path, "a", encoding="utf-8", newline="\n")

    def _open_fresh(self, tries=500):
        """v1.2: 找一个"还不存在"的文件号新建。
        受限环境里"写已存在文件"可能被拒(Errno 9), 但"新建"永远可行 ——
        所以续写失败时顺延一个空号, 至少不丢数据、也不中断。"""
        for _ in range(tries):
            self.path = os.path.join(self.out_dir, "语料_%03d.txt" % self.idx)
            if not os.path.exists(self.path):
                self.cur = 0
                self.fh = open(self.path, "a", encoding="utf-8", newline="\n")
                return self.path
            self.idx += 1
        raise OSError("找不到可新建的语料文件号(试了 %d 个都被占了)" % tries)

    def _close_quiet(self):
        try:
            self.fh.flush()
            os.fsync(self.fh.fileno())
        except Exception:
            pass
        try:
            self.fh.close()
        except Exception:
            pass
        self.fh = None

    def _roll(self):
        self._close_quiet()
        self.idx += 1
        self._open()
        return self.path

    def write(self, lines):
        """v1.2: 写坏了不退赛 —— 老文件句柄失效时顺延到新文件, 本批重写一遍。"""
        with self.lock:
            try:
                return self._emit(lines)
            except OSError as e:
                log("!! 续写 %s 失败(%s), 顺延到新文件重写本批"
                    % (os.path.basename(self.path), e))
                self._close_quiet()
                self.idx += 1
                self._open_fresh()
                return self._emit(lines)

    def _emit(self, lines):
        rolled = None
        n = 0
        added = []
        db = 0
        try:
            for s in lines:
                k = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                if k in self.dedupe:
                    continue
                self.dedupe.add(k)
                added.append(k)
                blob = s + "\n"
                nb = len(blob.encode("utf-8"))
                self.fh.write(blob)
                self.cur += nb
                self.total += nb
                self.lines += 1
                db += nb
                n += 1
            self.fh.flush()
            if self.cur >= self.limit:
                rolled = self._roll()
            return n, rolled
        except Exception:
            # 这一批写废了 -> 字节数/行数/去重集全部回滚, 交给上层换新文件重写
            for k in added:
                self.dedupe.discard(k)
            self.cur -= db
            self.total -= db
            self.lines -= n
            raise

    def done(self):
        return self.total >= self.target

    def mb(self):
        return self.total / 1048576.0

    def close(self):
        with self.lock:
            if self.fh is not None:
                self._close_quiet()


# ============================================================================
# 4. 抓取 + 解析
# ============================================================================

def decode(blob):
    for enc in ("utf-8", "gb18030", "utf-8-sig", "big5", "latin-1"):
        try:
            t = blob.decode(enc)
            if t and "\ufffd" not in t[:3000]:
                return t
        except Exception:
            continue
    return blob.decode("utf-8", "ignore")


_TLS = threading.local()


def session():
    s = getattr(_TLS, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _TLS.s = s
    return s


def fetch(url):
    """返回 HTML 文本(最多 MAX_HTML 字节); 失败抛异常."""
    r = session().get(url, timeout=TIMEOUT, verify=False, allow_redirects=True,
                      stream=True)
    try:
        if r.status_code >= 400:
            raise IOError("HTTP %d" % r.status_code)
        ctype = (r.headers.get("Content-Type") or "").lower()
        if ("text/html" not in ctype) and ("xml" not in ctype) and ctype:
            raise IOError("非 HTML: %s" % ctype[:40])
        blob = r.raw.read(MAX_HTML, decode_content=True)
    finally:
        r.close()
    if not blob:
        raise IOError("空响应")
    return decode(blob)


KILL_TAGS = ("script", "style", "noscript", "iframe", "svg", "canvas",
             "form", "nav", "footer", "header", "aside", "select",
             "button", "option", "figure", "video", "audio")


def extract(html):
    """返回 (优质行列表, 候选链接列表). 解析完立刻丢 soup, 不驻留. """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(KILL_TAGS):
        tag.decompose()

    links = []
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        links.append(href)

    try:
        raw = soup.get_text("\n", strip=True)
    finally:
        soup.decompose()

    lines = raw.split("\n")
    del raw
    return page_pick(lines), links


# ============================================================================
# 5. 链接筛选
# ============================================================================

_MULTI = ("com.cn", "gov.cn", "org.cn", "net.cn", "edu.cn", "ac.cn",
          "co.jp", "co.uk", "com.hk", "com.tw", "com.sg")


def root_domain(netloc):
    host = netloc.split("@")[-1].split(":")[0].lower()
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    last2 = ".".join(parts[-2:])
    if last2 in _MULTI:
        return ".".join(parts[-3:])
    return last2


def looks_content(path):
    if NUMRUN_RE.search(path):
        return True
    low = path.lower()
    return low.endswith((".html", ".htm", ".shtml", ".jsp", ".asp", ".aspx"))


def usable_link(url, seed_root, depth, seed_url):
    try:
        sp = urlsplit(url)
    except Exception:
        return False
    if sp.scheme not in ("http", "https"):
        return False
    if not sp.netloc:
        return False
    if root_domain(sp.netloc) != seed_root:
        return False
    path = sp.path or "/"
    low = path.lower()
    for ext in BAD_EXT:
        if low.endswith(ext):
            return False
    for w in JUNK_PATH:
        if w in low:
            return False
    if len(url) > 220 or len(path) > 160:
        return False
    if path.count("/") > 7:
        return False
    if depth == 0:
        return True
    # v1.1: 队列老见底, 是因为"更深的必须长得像正文页"这条筛得太狠 ——
    #   栏目页 / 分页列表 / 专题聚合页全被挡在外面, 于是爬两下就没得爬了。
    #   现在: 深度 3 以内的同域链接先收进来(是不是正文交给 page_pick 判, 垃圾页
    #   只会白跑一趟, 不会污染语料); 再深才要求"像正文页"。
    if depth <= 3:
        return True
    return looks_content(path)


# ============================================================================
# 6. 主流程
# ============================================================================

def log(msg):
    sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def reseed(frontier, seen, round_no):
    """队列见底时的补种(v1.1 —— 之前就是这里缺了一口气, 22MB 就收工了):

    ① 种子池里还没爬过的(站点 × 栏目)重新灌一遍;
    ② PAGERS 的分页列表页, 每轮往后推 40 页(第 2..41, 42..81, ...)。
    返回本轮实际灌进去的条数; 返回 0 表示真没得爬了。
    """
    added = []
    for name, url in SEEDS:
        if not seen.has(url):
            added.append((url, 0, name))
    lo = 2 + round_no * 40
    for name, tpl in PAGERS:
        for n in range(lo, lo + 40):
            try:
                u = tpl % n
            except Exception:
                continue
            if not seen.has(u):
                added.append((u, 1, name))
    if not added:
        return 0
    random.shuffle(added)
    take = added[:max(0, FRONTIER_MAX - len(frontier))]
    frontier.extend(take)
    return len(take)


def run(args):
    out_dir = os.path.abspath(args.out)
    if args.state_path:
        state_dir = os.path.abspath(args.state_path)
    else:
        state_dir = os.path.join(out_dir, STATE_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    state_path = os.path.join(state_dir, "state.json")
    seen_path = os.path.join(state_dir, "seen.txt")

    if args.reset_state:
        for p in (state_path, seen_path):
            if os.path.exists(p):
                os.remove(p)
        log("断点已清空, 从头开始")

    st = load_state(state_path)
    seen = Seen(seen_path)

    # v1.2: 叠加"只读历史" —— 换新状态目录时, 老 seen.txt 里 1 万多个已爬指纹不能丢,
    #   否则会把老页面重爬一遍(还会重复写语料)。读了只是内存去重, 不写回去。
    for ep in (args.seen_extra or []):
        try:
            cnt = 0
            with open(ep, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    k = line.strip()
                    if k:
                        seen.done.add(k)
                        cnt += 1
            log("叠加历史: %s -> %d 条 (合计 %d)" % (os.path.basename(ep), cnt, len(seen.done)))
        except Exception as e:
            log("叠加历史失败: %s (%s)" % (ep, e))

    # 断点有效性: 产出目录里已存在的语料字节数
    base = 0
    try:
        for n in os.listdir(out_dir):
            p = os.path.join(out_dir, n)
            if n.lower().endswith(".txt") and not n.startswith(("_", ".")) \
                    and os.path.isfile(p):
                base += os.path.getsize(p)
    except Exception:
        pass

    start_idx = int(args.start_index) if args.start_index else int(st.get("file_index", 1))
    writer = Writer(out_dir, args.file_mb, args.target_mb, start_idx, base)
    src_pages = dict(st.get("src_pages") or {})
    pages = int(st.get("pages") or 0) or int(args.base_pages or 0)

    frontier = deque()
    for item in (st.get("frontier") or []):
        try:
            u, d, s = item[0], int(item[1]), item[2]
        except Exception:
            continue
        if not seen.has(u):
            frontier.append((u, d, s))
    if not frontier:
        for name, url in SEEDS:
            if not seen.has(url):
                frontier.append((url, 0, name))

    log("=" * 66)
    log("产出目录 : %s" % out_dir)
    log("状态目录 : %s   (下划线开头, 不会被当语料)" % state_dir)
    log("目标     : %.0f MB (单文件 %.0f MB) | 已存在 %.2f MB | 断点页数 %d"
        % (args.target_mb, args.file_mb, base / 1048576.0, pages))
    log("队列     : %d 条 | 历史已爬 %d 个 URL | 线程 %d | 深度 %d"
        % (len(frontier), len(seen.done), args.workers, args.depth))
    log("=" * 66)

    t0 = time.time()
    last_ck = [time.time()]
    ok_pages = [0]
    new_lines = [0]
    lock = threading.Lock()
    seed_round = [0]                  # 已经补种了几轮

    def checkpoint(force=False):
        if (not force) and (time.time() - last_ck[0] < 20.0):
            return
        last_ck[0] = time.time()
        seen.flush()
        keep = []
        with lock:
            items = list(frontier)
        if len(items) > 60000:
            items = items[:60000]
        for u, d, s in items:
            keep.append([u, d, s])
        save_state(state_path, {
            "version": STATE_VERSION,
            "frontier": keep,
            "file_index": writer.idx,
            "pages": pages,
            "written": writer.total,
            "src_pages": src_pages,
        })
        log("  [存] 页 %d | 语料 %.2f MB | 优质句 %d | 队列 %d | 本文件 %s"
            % (pages, writer.mb(), writer.lines,
               len(items), os.path.basename(writer.path)))

    def worker(url, src):
        html = fetch(url)
        lines, links = extract(html)
        del html
        return lines, links

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            while not writer.done():
                if args.minutes and (time.time() - t0) > args.minutes * 60:
                    log("到点(%.0f 分钟), 收工" % args.minutes)
                    break

                batch = []
                with lock:
                    while frontier and len(batch) < args.workers * 3:
                        url, depth, src = frontier.popleft()
                        if src_pages.get(src, 0) >= args.pages_per_source:
                            continue
                        if not seen.claim(url):
                            continue
                        batch.append((url, depth, src))
                if not batch:
                    if frontier:
                        continue
                    # v1.1: 没到目标 MB 不能收工 —— 先补种(种子池 + 分页模板),
                    #   补得出来就接着爬, 补不出来了才认输。老版本就是这里直接
                    #   break, 所以 22MB 就"卡住不动"了。
                    if not writer.done() and seed_round[0] < RESEED_ROUNDS:
                        n = reseed(frontier, seen, seed_round[0])
                        seed_round[0] += 1
                        if n:
                            log("[补种] 第 %d 轮, 队列重新灌入 %d 条 (已 %.2f MB / %.0f MB)"
                                % (seed_round[0], n, writer.mb(), args.target_mb))
                            continue
                    log("队列空了, 收工")
                    break

                futs = {}
                for (url, depth, src) in batch:
                    futs[ex.submit(worker, url, src)] = (url, depth, src)

                for fu in as_completed(futs):
                    url, depth, src = futs[fu]
                    try:
                        lines, links = fu.result()
                    except Exception:
                        lines, links = [], []
                        links = []
                    seen.finish(url)
                    with lock:
                        pages += 1
                        src_pages[src] = src_pages.get(src, 0) + 1

                    if lines:
                        try:
                            n, rolled = writer.write(lines)
                        except Exception as e:
                            # v1.2: 单批写盘失败不该中止整个爬取 —— 记一笔, 继续跑
                            log("!! 写语料失败(%s), 跳过这一批继续爬" % e)
                            n, rolled = 0, None
                        new_lines[0] += n
                        ok_pages[0] += 1
                        if n and pages % 20 < 3:
                            log("  页 %d | %s | +%d 句 | 共 %.2f MB"
                                % (pages, src, n, writer.mb()))
                        if rolled:
                            log("  [滚] 新文件 %s" % os.path.basename(rolled))

                    if depth < args.depth and links:
                        seed_root = root_domain(urlsplit(url).netloc)
                        add = []
                        for lk in links:
                            u = urljoin(url, lk)
                            u = u.split("#")[0].strip()
                            if not u:
                                continue
                            if usable_link(u, seed_root, depth + 1, url):
                                if not seen.has(u):
                                    add.append((u, depth + 1, src))
                        random.shuffle(add)
                        with lock:
                            if add:
                                seed_round[0] = 0      # 又扒出新链接了 -> 补种计数归零
                            if len(frontier) < FRONTIER_MAX:
                                for it in add:
                                    frontier.append(it)

                    if pages % 50 == 0:
                        checkpoint()
                        log("  进度 页 %d | %.2f MB / %.0f MB | 优质句 %d | 队列 %d"
                            % (pages, writer.mb(), args.target_mb,
                               writer.lines, len(frontier)))
                        if writer.done():
                            break

                checkpoint()
    except KeyboardInterrupt:
        log("收到 Ctrl+C, 正在保存断点 ...")
    finally:
        checkpoint(force=True)
        writer.close()
        seen.flush()

    log("=" * 66)
    log("结束: 爬了 %d 页 | 优质句 %d | 语料 %.2f MB | 耗时 %.1f 分钟"
        % (pages, writer.lines, writer.mb(), (time.time() - t0) / 60.0))
    log("下一个 断点/续跑: 重新运行本脚本即可, 会从游标处接着爬.")
    log("=" * 66)


def main():
    ap = argparse.ArgumentParser(description="爬取优质中文语料(可断点续跑)")
    ap.add_argument("--out", default=OUT_DIR, help="产出目录")
    ap.add_argument("--target-mb", type=float, default=50.0, help="累计目标 MB")
    ap.add_argument("--file-mb", type=float, default=5.0, help="单文件滚动 MB")
    ap.add_argument("--workers", type=int, default=12, help="并发线程")
    ap.add_argument("--depth", type=int, default=4, help="跟随深度")
    ap.add_argument("--minutes", type=float, default=0.0, help="最长运行分钟, 0=不限")
    ap.add_argument("--pages-per-source", type=int, default=6000,
                    help="单个栏目最多爬多少页")
    ap.add_argument("--reset-state", action="store_true", help="清空断点重来")
    # v1.2: 断点目录可另指 / 叠加只读历史 / 指定起始文件号 —— 用于"产出目录只能读、不能改老文件"
    #   的场景(例如在受限沙箱里续跑老语料): 断点挪到可写盘, 老 seen.txt 当只读历史叠加进来。
    ap.add_argument("--state-path", default="",
                    help="断点目录(绝对路径); 留空=<产出目录>/%s" % STATE_DIRNAME)
    ap.add_argument("--seen-extra", action="append", default=[],
                    help="只读历史 seen 文件(可多次), 只用于内存去重")
    ap.add_argument("--start-index", type=int, default=0, help="语料起始文件号, 0=跟断点")
    ap.add_argument("--base-pages", type=int, default=0, help="页面计数起始值(仅日志好看用)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
