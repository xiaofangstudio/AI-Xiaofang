# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xiaofang_v06alpha as XF

# 冒烟测试: 关闭深度思考打字显示以提速回归, 事后再恢复用户偏好
_XF_OLD_SHOW = XF.SHOW_DEEP_THINK
XF.SHOW_DEEP_THINK = False
XF.THINK_MODE = "off"

print("=== 实例化模型 (硬件自适应) ===")
xf = XF.XiaoFang()
print("版本     =", XF.VERSION)
print("引擎名   =", XF.ENGINE_NAME)
print("CPU      =", XF.HW["cpu"], " 核数 =", XF.HW["cores"])
print("显卡     =", XF.HW["gpu_model"], " 显存 =", XF.HW["vram"], "MB")
print("内存    总 =", XF.HW["mem_total"], " 空闲 =", XF.HW["mem_free"])
print("GPU后端  =", XF.HAS_GPU)
print("d_model  =", xf.transformer.d_model)
print("n_layers =", xf.transformer.n_layers)
print("n_heads  =", xf.transformer.n_heads)
print("vocab    =", xf.transformer.vocab_size)
print("KB docs  =", len(xf.retriever.docs))
print("思考模式 =", XF.THINK_MODE, " 多轮数 =", XF.THINK_TURNS, " 情绪敏感度 =", XF.EMO_SENS)
if hasattr(xf, "presets"):
    print("预设数   =", len(xf.presets))
if hasattr(xf, "learner"):
    print("自升级大脑 = 已启用")

# v0.6: 冒烟测试本身走"不思考"快速通道; 单测思考/多轮通过直接调用验证即可
tests = [
    "你是谁", "你是什么模型",
    "小方工作室官网是什么",
    "帮我介绍下Python",
    "量子引力最新研究是什么",
    "我好累啊 😭",                       # emoji 情绪
    "今天的日期是几号",
    "hello", "こんにちは", "안녕하세요", "bonjour",
    "现在几点钟了", "给我讲个笑话", "谢谢你了",
    "什么是黑洞",
    "你家住在哪里", "你几岁啦",
    "什么是编程",
    "35乘7等于多少", "256除以4等于多少",
    "帮我用冒泡排序给数组排序", "写一个阶乘函数",
    "什么是向量机",
]
for t in tests:
    print("\n\n========== 输入: %s ==========" % t)
    try:
        xf.process_chat(t)
    except Exception:
        import traceback
        traceback.print_exc()

# v0.6: 单独验证多轮思考递送不抛异常
print("\n=== 验证多轮思考 (multi) ===")
XF.THINK_MODE = "multi"
XF.SHOW_DEEP_THINK = True
try:
    xf.process_chat("给我解释一下自动驾驶")
except Exception:
    import traceback
    traceback.print_exc()
XF.THINK_MODE = "think"
XF.SHOW_DEEP_THINK = _XF_OLD_SHOW

# 验证懒加载后的 score_distribution 候选集规模(不再遍历上万词库)
print("\n=== 懒加载验证 ===")
emo = XF.EmotionAnalyzer(xf.tokenizer).analyze("什么是黑洞")
seed = xf.tokenizer.tokenize("什么是黑洞")
dist = xf.transformer.score_distribution(seed, bias_tokens=set(seed), probs=None)
print("候选集 tok 数 =", len(dist), "(远小于 vocab<={})".format(xf.transformer.vocab_size))

print("\n=== 冒烟测试结束 ===")
print("累计自学习: 生词 = %s, 知识 = %s" % xf.learner.report())