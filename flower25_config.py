"""25 类实验的类别范围；保留原五类的编号，新增二十类。"""

BASE_CLASSES = [
    ("daisy", "雏菊"), ("dandelion", "蒲公英"), ("roses", "玫瑰"),
    ("sunflowers", "向日葵"), ("tulips", "郁金香"),
]

# 官方类别名来自 Oxford Flowers102，不把所有带 lily 的名称都并入百合。
# 百合样本限于 tiger lily；兰花与鸢尾分别合并下面明确列出的来源类别。
EXTRA_CLASSES = [
    ("lilies", "百合（卷丹百合）", ["tiger lily"]),
    ("lotus", "荷花", ["lotus"]),
    ("orchids", "兰花（选取三类）", [
        "moon orchid", "hard-leaved pocket orchid", "ruby-lipped cattleya",
    ]),
    ("camellias", "山茶花", ["camellia"]),
    ("hibiscus", "木槿属花卉", ["hibiscus"]),
    ("carnations", "康乃馨", ["carnation"]),
    ("daffodils", "黄水仙", ["daffodil"]),
    ("irises", "鸢尾（选取两类）", ["bearded iris", "yellow iris"]),
    ("california_poppies", "花菱草", ["californian poppy"]),
    ("petunias", "矮牵牛", ["petunia"]),
    ("bougainvillea", "三角梅", ["bougainvillea"]),
    ("magnolias", "木兰属花卉", ["magnolia"]),
    ("marigolds", "万寿菊", ["marigold"]),
    ("water_lilies", "睡莲", ["water lily"]),
    ("bird_of_paradise", "鹤望兰", ["bird of paradise"]),
    ("frangipani", "鸡蛋花", ["frangipani"]),
    ("anthuriums", "红掌", ["anthurium"]),
    ("azaleas", "杜鹃花", ["azalea"]),
    ("cyclamens", "仙客来", ["cyclamen"]),
    ("snapdragons", "金鱼草", ["snapdragon"]),
]

CLASS_NAMES = [name for name, _ in BASE_CLASSES] + [row[0] for row in EXTRA_CLASSES]
CLASS_NAMES_ZH = dict(BASE_CLASSES)
CLASS_NAMES_ZH.update({name: chinese for name, chinese, _ in EXTRA_CLASSES})
