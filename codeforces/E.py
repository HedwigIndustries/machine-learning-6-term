import math
from collections import Counter


def main():
    _, _ = read_k()
    n, pairs = read_pairs()
    res = calc_entropy(n, pairs)
    print(-res)


def read_k():
    data = input().split()
    k_x = int(data[0])
    k_y = int(data[1])
    return k_x, k_y


def read_pairs():
    n = int(input())
    pairs = []
    for i in range(n):
        x, y = input().split()
        pairs.append((int(x), int(y)))
    return n, pairs


def calc_entropy(n, pairs):
    pairs_cnt = Counter(pairs)
    x_cnt = Counter([x for (x, _) in pairs])
    entropy = 0.0
    for (x, y), cnt in pairs_cnt.items():
        p_x = x_cnt[x] / n
        p_y_if_x = cnt / x_cnt[x]
        if p_y_if_x > 0:
            entropy += p_x * p_y_if_x * math.log(p_y_if_x)
    return entropy


if __name__ == '__main__':
    main()
