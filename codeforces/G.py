def read_objects(n):
    pairs = [[int(item) for item in input().split()] for _ in range(n)]
    obj_1 = [pair[0] for pair in pairs]
    obj_2 = [pair[1] for pair in pairs]
    return obj_1, obj_2


def init_ranks(obj_1, obj_2):
    x1_rank = {x: i + 1 for i, x in enumerate(sorted(obj_1))}
    x2_rank = {x: i + 1 for i, x in enumerate(sorted(obj_2))}
    return x1_rank, x2_rank


def get_actual_ranks(obj_1, obj_2, x1_rank, x2_rank):
    ranks_x1 = [x1_rank[v] for v in obj_1]
    ranks_x2 = [x2_rank[v] for v in obj_2]
    return ranks_x1, ranks_x2


def calc_rho(n, square_ranks_diff):
    if (denumenator := n * (n ** 2 - 1)) == 0:
        return 0
    diff_ = 6 * square_ranks_diff
    return 1 - diff_ / denumenator


def main():
    n = int(input())
    obj_1, obj_2 = read_objects(n)
    x1_rank, x2_rank = init_ranks(obj_1, obj_2)
    ranks_x1, ranks_x2 = get_actual_ranks(obj_1, obj_2, x1_rank, x2_rank)

    square_ranks_diff = sum((ranks_x1[i] - ranks_x2[i]) ** 2 for i in range(n))
    rho = calc_rho(n, square_ranks_diff)
    print(rho)


if __name__ == '__main__':
    main()
