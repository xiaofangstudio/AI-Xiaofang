# 示例产物 · 贪吃蛇（纯标准库 tkinter，无第三方依赖）
# 用法：python 示例_贪吃蛇.py
# 方向键控制；吃到红果 +1 分、蛇 +1；撞墙或咬到自己结束。
import random
import tkinter as tk

CELL = 22
W, H = 24, 18          # 网格 24×18
STEP_MS = 130          # 步进节奏

class Snake:
    def __init__(self, canvas):
        self.cv = canvas
        self.reset()

    def reset(self):
        cx, cy = W // 2, H // 2
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.dir = (1, 0)
        self.dead = False
        self.score = 0
        self.place_food()

    def place_food(self):
        free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in self.body]
        self.food = random.choice(free) if free else None

    def turn(self, dx, dy):
        if (dx, dy) != (-self.dir[0], -self.dir[1]):   # 不许瞬时掉头
            self.dir = (dx, dy)

    def step(self):
        if self.dead:
            return
        head = (self.body[0][0] + self.dir[0], self.body[0][1] + self.dir[1])
        if (not (0 <= head[0] < W and 0 <= head[1] < H)) or head in self.body:
            self.dead = True
            return
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.body.pop()

    def draw(self):
        self.cv.delete('all')
        for i, (x, y) in enumerate(self.body):
            c = '#3b56e8' if i == 0 else '#6b86ff'    # 头最深
            self.cv.create_rectangle(x * CELL + 1, y * CELL + 1,
                                     x * CELL + CELL, y * CELL + CELL,
                                     fill=c, outline='')
        if self.food:
            fx, fy = self.food
            self.cv.create_oval(fx * CELL + 3, fy * CELL + 3,
                                fx * CELL + CELL - 3, fy * CELL + CELL - 3,
                                fill='#e5484d', outline='')

root = tk.Tk()
root.title('贪吃蛇 · 产物示例')
canvas = tk.Canvas(root, width=W * CELL, height=H * CELL, bg='#eef0f4',
                   highlightthickness=0)
canvas.pack(padx=12, pady=12)
info = tk.Label(root, text='得分 0 · 方向键控制 · Esc 重开', font=('Consolas', 11))
info.pack()

game = Snake(canvas)

def tick():
    game.step()
    game.draw()
    info.config(
        text=('得分 %d · 撞了，按 Esc 重开' % game.score) if game.dead
        else ('得分 %d · 方向键控制' % game.score))
    root.after(STEP_MS, tick)

def on_key(e):
    k = e.keysym
    if k == 'Up':    game.turn(0, -1)
    elif k == 'Down': game.turn(0, 1)
    elif k == 'Left': game.turn(-1, 0)
    elif k == 'Right': game.turn(1, 0)
    elif k == 'Escape':
        game.reset()
        game.draw()

root.bind('<Key>', on_key)
root.focus_set()
game.draw()
tick()
root.mainloop()