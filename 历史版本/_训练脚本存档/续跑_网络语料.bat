@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  网络语料爬虫 - 续跑 (双击即可, 自动从断点接着爬)
echo  产出:  C:\Users\ASUS\Desktop\小方小说\网络语料\语料_*.txt
echo  断点:  C:\Users\ASUS\Desktop\AI 小方\.crawl_status\state.json
echo  目标:  50 MB (单文件满 5 MB 自动新建下一个)
echo  关掉本窗口 / 合盖 = 停止; 重新双击 = 从断点继续
echo ============================================================
echo.
python "%~dp0爬取语料.py" --target-mb 50 --state-path "%~dp0.crawl_status" --seen-extra "c:\Users\ASUS\Desktop\小方小说\网络语料\_crawl\seen.txt"
echo.
echo 已退出。按任意键关闭窗口。
pause >nul
