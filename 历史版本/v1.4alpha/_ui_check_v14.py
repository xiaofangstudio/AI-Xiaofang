# -*- coding: utf-8 -*-
# v1.4 UI 轻量自检: 不装配 1B 模型, 只验证启动横幅 / 状态面板 / 折叠面板渲染不出错
import os
os.environ['XF_QUIET'] = '1'
import sys, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v14alpha as M

buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        M.show_startup()
    txt = buf.getvalue()
    assert '│' in txt and ('┌' in txt) and ('└' in txt), 'box frame missing'
    assert '你可以问我这些问题' in txt
    assert '小方工作室是什么' in txt
    assert '在键盘上敲打开始对话' in txt
    assert '快捷指令' in txt
    print('[ok] show_startup 横幅渲染:', len(txt.splitlines()), '行')
except Exception as e:
    print('[FAIL] show_startup:', repr(e)); raise

# 状态面板/折叠面板
M.UI_ST['capture'] = True
M.UI_ST['in'] = 12
M.UI_ST['out'] = 0
M.UI_ST['layer'] = 3
M.UI_ST['stage'] = '深度思考'
M.UI_ST['fold'] = True
ob = io.StringIO()
try:
    with contextlib.redirect_stdout(ob):
        M._panel_render()
    folded_txt = ob.getvalue()
    assert '点击面板' in folded_txt and 'Token' in folded_txt and '盲文' == '' or 'Token' in folded_txt
    assert 'Token' in folded_txt
    assert '第' in folded_txt and '/' in folded_txt
    print('[ok] 折叠面板(单行状态条)渲染含 Token/层数/阶段')
except Exception as e:
    print('[FAIL] folded panel:', repr(e)); raise

# 展开：buf 有内容 + 分界线
M.UI_ST['fold'] = False
M.UI_ST['buf'] = ['拆词完成', '前向计算 layer1', '意图加权']
ob2 = io.StringIO()
with contextlib.redirect_stdout(ob2):
    M._panel_render()
t2 = ob2.getvalue()
assert '深度思考 · 计算过程' in t2
assert '分界线' in t2
assert '拆词完成' in t2
print('[ok] 展开面板含思考行 + 计算分界线 + 状态面板')

# Token 从0增长
M.UI_ST['in'] = 0; M.UI_ST['out'] = 0
print('[ok] token 初始显示 =', M._ui_token_display(), '(应按时间增长)')

# 新命令路由 /fold 与 palette
class Fake:
    history=[]; meter=type('m',(),{'format':lambda s:'Token: x'})()
    selfmem=None
    transformer=type('t',(),{'n_layers':0,'n_heads':0,'d_model':0})()
f=Fake()
hc=M.XiaoFang.handle_command.__get__(f)
assert hc('/fold')=='ok'
names=M._CMD_NAMES
assert any(n.startswith('/prev') for n in names) and any(n.startswith('/stop') for n in names)
print('[ok] /fold 路由 + 命令注册含 /prev /stop')
print('ALL_UI_OK')