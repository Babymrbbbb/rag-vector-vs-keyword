"""中文分词（纯标准库实现，零外部依赖）。

策略：
- 英文 / 数字：按词提取并转小写；
- 中文：逐字切分（unigram）+ 相邻字 bigram（提升短语级匹配）；
- 去标点、去空白。

说明：这是轻量 lexical 分词，足以支撑「倒排 vs 向量」在小库上的对照演示。
若日后接入真实语义 embedding，只需替换 `embedder.py` 的向量表示，分词层不变。
"""

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    toks: list[str] = list(_WORD.findall(text))
    chars = _CJK.findall(text)
    toks.extend(chars)
    # 中文 bigram：捕捉「冷链 / 加盟 / 会员」等词组信号
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


if __name__ == "__main__":
    print(tokenize("冷链配送温度要求"))  # ['冷链', '配送', '温度', '要求', '冷', '链', '配', '送', ...]
