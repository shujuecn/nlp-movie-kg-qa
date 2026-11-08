# ruff: noqa
# 本文件是 docs/images/architecture.png 的生成脚本（绘图资产，不进包、不进 mypy 范围），
# 手动运行：python docs/images/draw_architecture.py；文字全部确定性渲染，杜绝错字。
"""确定性绘制 README 架构流程图（论文插图风格：虚线面板 + 马卡龙色块 + 空心箭头）。

文字用 Pillow + PingFang SC 渲染（像素级可控），布局尺寸动态计算。
产出：与本脚本同目录的 architecture.png（2048x880）。
"""

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 2048, 880
INK = "#111827"  # 正文黑
DASH_BLUE = "#7db8e8"  # 面板虚线框
BLUE_FILL = "#dbeafe"  # 浅蓝（本地）
BLUE_CHIP = "#bfdbfe"  # 蓝色标题条
GREEN_FILL = "#e3f4df"  # 浅绿（图谱构建）
ORANGE_FILL = "#ffedd5"  # 浅橙（在线服务）
ORANGE_CHIP = "#fed7aa"  # 橙色标题条/角标

FONT = "/System/Library/Fonts/PingFang.ttc"


def font(size, bold=False):
    # PingFang.ttc: index 0 = Regular，1 = Semibold（近似加粗）
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


img = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(img)
f_title = font(40, True)  # 节点主标题
f_sub = font(30)  # 节点小字
f_panel = font(44, True)  # 面板底部标题
f_chip = font(29, True)  # 分区标题条
f_legend = font(28)
f_badge = font(26, True)


def tw(s, f):
    return d.textlength(s, font=f)


def center_text(cx, y, s, f, fill=INK):
    d.text((cx - tw(s, f) / 2, y), s, font=f, fill=fill)


def dashed_rounded_rect(box, radius=20, dash=14, gap=9, width=3, color=DASH_BLUE):
    """沿圆角矩形路径画虚线（直边逐段 + 圆角隔段）。"""
    x0, y0, x1, y1 = box
    segs = [
        ((x0 + radius, y0), (x1 - radius, y0)),
        ((x1, y0 + radius), (x1, y1 - radius)),
        ((x1 - radius, y1), (x0 + radius, y1)),
        ((x0, y1 - radius), (x0, y0 + radius)),
    ]
    for p, q in segs:
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        t = 0.0
        while t < length:
            t2 = min(t + dash, length)
            d.line([(p[0] + ux * t, p[1] + uy * t), (p[0] + ux * t2, p[1] + uy * t2)],
                   fill=color, width=width)
            t = t2 + gap
    corners = [
        (x1 - radius, y0 + radius, 270, 360),
        (x1 - radius, y1 - radius, 0, 90),
        (x0 + radius, y1 - radius, 90, 180),
        (x0 + radius, y0 + radius, 180, 270),
    ]
    for cx, cy, a0, a1 in corners:
        prev = None
        for i in range(9):
            ang = math.radians(a0 + (a1 - a0) * i / 8)
            pt = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
            if prev is not None and (i - 1) % 2 == 0:
                d.line([prev, pt], fill=color, width=width)
            prev = pt


def arrow(p, q, width=4, head=16, color=INK):
    d.line([p, q], fill=color, width=width)
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    a1 = (q[0] - head * math.cos(ang - 0.42), q[1] - head * math.sin(ang - 0.42))
    a2 = (q[0] - head * math.cos(ang + 0.42), q[1] - head * math.sin(ang + 0.42))
    d.polygon([q, a1, a2], fill=color)


def dashed_arrow(p, q, width=3, dash=10, gap=7, head=13, color=INK):
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        d.line([(p[0] + ux * t, p[1] + uy * t), (p[0] + ux * t2, p[1] + uy * t2)],
               fill=color, width=width)
        t = t2 + gap
    for end in (q, p):
        ex, ey = end
        a1 = (ex - head * math.cos(0.42) * ux + head * math.sin(0.42) * uy,
              ey - head * math.cos(0.42) * uy - head * math.sin(0.42) * ux)
        a2 = (ex - head * math.cos(0.42) * ux - head * math.sin(0.42) * uy,
              ey - head * math.cos(0.42) * uy + head * math.sin(0.42) * ux)
        d.polygon([end, a1, a2], fill=color)


def hollow_arrow(x0, x1, cy, shaft=17, head_h=50, head_len=30, color=INK):
    """面板之间的空心大箭头（白底黑描边）。"""
    back = x1 - head_len
    pts = [
        (x0, cy - shaft), (back, cy - shaft), (back, cy - head_h / 2), (x1, cy),
        (back, cy + head_h / 2), (back, cy + shaft), (x0, cy + shaft),
    ]
    d.polygon(pts, fill="white")
    d.line(pts + [pts[0]], fill=color, width=3, joint="curve")


def rbox(box, fill, radius=18, outline=None, owidth=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline,
                        width=owidth if outline else 0)


# ---- 顶部图例条（动态排宽，避免文字与虚线箭头重叠）----
items = [("arrow", "数据流向"), ("blue", "本地处理"), ("green", "图谱构建"),
         ("orange", "在线服务"), ("dash", "组件交互")]
CHIP_W, GAP_ITEM = 84, 56
widths = [CHIP_W + 20 + tw(label, f_legend) for _, label in items]
total = sum(widths) + GAP_ITEM * (len(items) - 1)
lx = (W - total) / 2
d.rounded_rectangle((lx - 44, 24, lx + total + 44, 100), radius=22,
                    outline="#cbd5e1", width=2)
for (kind, label), item_w in zip(items, widths):
    cy = 62
    if kind == "arrow":
        arrow((lx, cy), (lx + 70, cy), width=4, head=16)
        lx += 70 + 20
    elif kind == "dash":
        dashed_arrow((lx, cy), (lx + 70, cy))
        lx += 70 + 20
    else:
        fill = {"blue": BLUE_FILL, "green": GREEN_FILL, "orange": ORANGE_FILL}[kind]
        rbox((lx, cy - 22, lx + CHIP_W, cy + 22), fill, radius=10)
        lx += CHIP_W + 20
    d.text((lx, cy - 18), label, font=f_legend, fill=INK)
    lx += tw(label, f_legend) + GAP_ITEM

# ---- 三个虚线面板 ----
TOP, BOT = 138, 800
p1 = (30, TOP, 550, BOT)
p2 = (632, TOP, 1195, BOT)
p3 = (1278, TOP, 2018, BOT)
for box in (p1, p2, p3):
    dashed_rounded_rect(box)
center_text((p1[0] + p1[2]) / 2, 716, "数据采集", f_panel)
center_text((p2[0] + p2[2]) / 2, 716, "图谱构建（备课期离线）", f_panel)
center_text((p3[0] + p3[2]) / 2, 716, "混合问答（在线）", f_panel)

# ---- 面板一：豆瓣 → movies.jsonl ----
b1 = (70, 190, 510, 350)
rbox(b1, BLUE_FILL)
center_text((b1[0] + b1[2]) / 2, 214, "豆瓣：周星驰电影专题", f_title)
center_text((b1[0] + b1[2]) / 2, 278, "122 条 → 入库 73 部", f_sub)
arrow((290, 350), (290, 406))
b2 = (70, 406, 510, 560)
rbox(b2, "#e4eaf7")
center_text((b2[0] + b2[2]) / 2, 430, "data/movies.jsonl", f_title)
center_text((b2[0] + b2[2]) / 2, 494, "全字段 + 简介", f_sub)

# ---- 面板二：build_kg / extract_llm 并行分叉 ----
g1 = (672, 175, 1155, 350)
rbox(g1, GREEN_FILL)
center_text((g1[0] + g1[2]) / 2, 199, "kgqa/build_kg.py", f_title)
center_text((g1[0] + g1[2]) / 2, 268, "确定性三元组 +", f_sub)
center_text((g1[0] + g1[2]) / 2, 306, "实体别名表", f_sub)

g2 = (672, 420, 1155, 620)
rbox(g2, GREEN_FILL)
badge_w = tw("silra API ①", f_badge) + 36
badge = (1155 - 24 - badge_w, 396, 1155 - 24, 444)
rbox(badge, ORANGE_CHIP, radius=12)
center_text((badge[0] + badge[2]) / 2, 406, "silra API ①", f_badge)
center_text((g2[0] + g2[2]) / 2, 456, "kgqa/extract_llm.py", f_title)
center_text((g2[0] + g2[2]) / 2, 524, "语义三元组：", f_sub)
center_text((g2[0] + g2[2]) / 2, 562, "角色-饰演者、题材", f_sub)

# movies.jsonl 分叉两条实线箭头（两节点之间不画箭头）
arrow((510, 462), (672, 262))
arrow((510, 492), (672, 516))

# ---- 面板三：本地 + silra 两个半区 ----
local = (1296, 175, 1630, 690)
rbox(local, BLUE_FILL, radius=20)
api = (1688, 175, 1990, 690)
rbox(api, ORANGE_FILL, radius=20)

chip_w = tw("本地运行（CPU 即可）", f_chip) + 40
lc_x = 1296 + (local[2] - local[0] - chip_w) / 2
rbox((lc_x, 200, lc_x + chip_w, 258), BLUE_CHIP, radius=12)
center_text(lc_x + chip_w / 2, 208, "本地运行（CPU 即可）", f_chip)

api_label = "silra API（qwen3.8-flash）"
chip_w2 = tw(api_label, font(26, True)) + 32
ac_x = 1688 + (api[2] - api[0] - chip_w2) / 2
rbox((ac_x, 200, ac_x + chip_w2, 256), ORANGE_CHIP, radius=12)
center_text(ac_x + chip_w2 / 2, 207, api_label, font(26, True))

for main, sub, y in [
    ("NetworkX", "图谱谓词查询", 296),
    ("FAISS", "双索引", 432),
    ("Gradio", "工作台", 568),
]:
    bx = (1322, y, 1604, y + 118)
    rbox(bx, "white", radius=14, outline="#e2e8f0", owidth=2)
    center_text((bx[0] + bx[2]) / 2, y + 12, main, f_title)
    center_text((bx[0] + bx[2]) / 2, y + 66, sub, f_sub)

for main, y, h in [("② 在线问题理解", 320, 108), ("③ 在线 RAG 生成", 476, 148)]:
    bx = (1718, y, 1960, y + h)
    rbox(bx, "white", radius=14, outline="#e2e8f0", owidth=2)
    center_text((bx[0] + bx[2]) / 2, y + 14, main, font(31, True))
center_text((1718 + 1960) / 2, 568, "（要求引用）", f_sub)

# 本地 ⇄ silra 双向虚线（组件交互）
dashed_arrow((1630, 440), (1688, 440))

# ---- 面板间空心大箭头（数据流向）----
hollow_arrow(556, 626, 430)
hollow_arrow(1201, 1272, 430)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
img.save(out)
print(f"saved {out} {img.size}")
