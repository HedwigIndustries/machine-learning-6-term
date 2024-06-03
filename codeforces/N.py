def create_class_lst(lst, m):
    class_lst = [[] for _ in range(m)]
    for pos, class_ in enumerate(lst):
        class_lst[class_].append(pos)
    return class_lst


def fill_parts(class_lst, k):
    ptr = 0
    parts = [[] for _ in range(k)]
    for class_ in class_lst:
        for item in class_:
            cur_part = ptr % k
            parts[cur_part].append(item + 1)
            ptr += 1
    return parts


def pretty_print(parts):
    for part in parts:
        print(len(part), end=" ")
        part.sort()
        for item in part:
            print(item, end=" ")
        print()


def main():
    n, m, k = map(int, input().split())
    lst = [int(x) - 1 for x in input().split()]
    class_lst = create_class_lst(lst, m)
    parts = fill_parts(class_lst, k)
    pretty_print(parts)


if __name__ == '__main__':
    main()
