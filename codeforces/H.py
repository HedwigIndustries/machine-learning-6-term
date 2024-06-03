def read_table():
    _, _ = (int(x) for x in input().split())
    n = int(input())
    pairs_counter = {}
    x1_counter = {}
    x2_counter = {}
    for _ in range(n):
        x1, x2 = map(int, input().split())
        x1 -= 1
        x2 -= 1
        pairs_counter[(x1, x2)] = pairs_counter.get((x1, x2), 0) + 1
        x1_counter[x1] = x1_counter.get(x1, 0) + 1
        x2_counter[x2] = x2_counter.get(x2, 0) + 1
    return pairs_counter, x1_counter, x2_counter, n


def calc_hi_square(pairs_counter, x1_counter, x2_counter, n):
    hi_square = n
    for pair, value in pairs_counter.items():
        x = pair[0]
        y = pair[1]
        row_sum = x1_counter[x] / n
        col_sum = x2_counter[y] / n
        expected = n * row_sum * col_sum
        hi_square -= expected - ((value - expected) ** 2) / expected

    return hi_square


def main():
    pairs_counter, x1_counter, x2_counter, n = read_table()
    hi_square = calc_hi_square(pairs_counter, x1_counter, x2_counter, n)
    print(hi_square)


if __name__ == '__main__':
    main()
