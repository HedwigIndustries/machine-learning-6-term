from scipy.stats import pearsonr


def read_data():
    n, k = map(int, input().split())
    a = [int(x) for x in input().split()]
    b = [int(x) for x in input().split()]
    return a, b, k, n


def calculate_weighted_pearson(a, b, k, n):
    w_sum = 0
    total = 0

    for category in range(1, k + 1):
        indices = [i for i, x in enumerate(a) if x == category]
        if not indices:
            continue

        one_hot = [1 if a[i] == category else 0 for i in range(n)]

        corr, _ = pearsonr(one_hot, b)
        weight = len(indices)

        w_sum += corr * weight
        total += weight

    if total == 0:
        return 0

    return w_sum / total


def main():
    a, b, k, n = read_data()
    coef = calculate_weighted_pearson(a, b, k, n)
    print(f"{coef:.9f}")


if __name__ == "__main__":
    main()
