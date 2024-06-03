def calc_distances(k, n, pairs):
    x_lst = []
    sign_y_lst = [[] for _ in range(k)]
    for x, y in pairs:
        x_lst.append(x)
        sign_y_lst[y].append(x)
    x_lst.sort()

    intra_distance = calc_intra_distance(sign_y_lst)
    inter_distance = calc_inter_distance(x_lst, intra_distance)

    print(intra_distance)
    print(inter_distance)


def calc_intra_distance(x2x1):
    distance = 0
    for x in x2x1:
        x.sort()
        distance += recalc(x)
    return distance


def calc_inter_distance(x, intra_distance):
    distance = recalc(x)
    return distance - intra_distance


def recalc(x):
    distance = 0
    size = len(x)
    for i in range(size):
        b = size - 1
        distance += 2 * x[i] * (2 * i - b)
    return distance


def main():
    k = int(input())
    n = int(input())
    pairs = [[int(x) - 1 for x in input().split()] for _ in range(n)]
    calc_distances(k, n, pairs)


if __name__ == '__main__':
    main()
