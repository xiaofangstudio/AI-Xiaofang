# -*- coding: utf-8 -*-
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
        _buf.extend(line.rstrip("\r\n"))
        _buf.append("\r")          # 一行读满 = 用户按了回车
        return True

    def getwch():
        if not _buf and not _pull():
            return "\x03"          # EOF -> 引擎自己会当成中断干净退出
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
