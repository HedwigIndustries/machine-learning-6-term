from collections import Counter


def calc_conditional_variance(pairs):
    count = Counter([x for x, y in pairs])
    square_dct, sum_dct = init_dcts(count)
    for x, y in pairs:
        sum_dct[x] += y
        square_dct[x] += y * y
    total_variance = 0
    for x in sum_dct:
        variance_y = (square_dct[x] - sum_dct[x] ** 2 / count[x]) / count[x]
        total_variance += variance_y * count[x]
    total_count = sum(count.values())
    conditional_variance = total_variance / total_count
    return conditional_variance


def init_dcts(counts):
    sum_dct = {x: 0 for x in set(counts.keys())}
    squares_dct = {x: 0 for x in set(counts.keys())}
    return squares_dct, sum_dct


def main():
    _ = int(input())
    n = int(input())
    pairs = [[int(x) for x in input().split()] for _ in range(n)]
    conditional_variance = calc_conditional_variance(pairs)
    print(conditional_variance)


if __name__ == '__main__':
    main()
