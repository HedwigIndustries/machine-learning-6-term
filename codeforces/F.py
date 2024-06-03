def calc_measures(k, confusion_matrix):
    precision, recall, w_precision, w_recall, items_count_in_class, total_items = calc_values(k, confusion_matrix)

    micro_f1 = calc_f1(w_precision, w_recall)

    macro_f1 = calc_macro_f1(k, precision, recall, items_count_in_class, total_items)

    weighted_f1 = calc_weighted_f1(k, precision, recall, items_count_in_class, total_items)

    return micro_f1, macro_f1, weighted_f1


def calc_values(k, confusion_matrix):
    true_positive = []
    false_positive = []
    false_negative = []
    for i in range(k):
        tmp = confusion_matrix[i][i]
        true_positive.append(tmp)
        false_positive.append(sum(confusion_matrix[j][i] for j in range(k)) - tmp)
        false_negative.append(sum(confusion_matrix[i][j] for j in range(k)) - tmp)

    items_count_in_class = [true_positive[i] + false_negative[i] for i in range(k)]

    precision, recall = calc_precision_recall(
        k,
        true_positive,
        false_positive,
        false_negative
    )
    w_precision, w_recall = calc_w_precision_recall(
        k,
        true_positive,
        false_positive,
        false_negative,
        items_count_in_class
    )
    total_items = sum(items_count_in_class)
    return precision, recall, w_precision, w_recall, items_count_in_class, total_items


def calc_precision_recall(k, true_positive, false_positive, false_negative):
    precision = []
    recall = []
    for c in range(k):
        if true_positive[c] + false_positive[c] != 0:
            precision.append(true_positive[c] / (true_positive[c] + false_positive[c]))
        else:
            precision.append(0)
        if true_positive[c] + false_negative[c] != 0:
            recall.append(true_positive[c] / (true_positive[c] + false_negative[c]))
        else:
            recall.append(0)
    return precision, recall


def calc_w_precision_recall(k, true_positive, false_positive, false_negative, items_count_in_class):
    w_true_positive = sum([true_positive[i] * items_count_in_class[i] for i in range(k)])
    w_false_positive = sum([false_positive[i] * items_count_in_class[i] for i in range(k)])
    w_false_negative = sum([false_negative[i] * items_count_in_class[i] for i in range(k)])
    if w_true_positive + w_false_positive != 0:
        w_precision = w_true_positive / (w_true_positive + w_false_positive)
    else:
        w_precision = 0

    if w_true_positive + w_false_negative != 0:
        w_recall = w_true_positive / (w_true_positive + w_false_negative)
    else:
        w_recall = 0
    return w_precision, w_recall


def calc_weighted_f1(k, precision, recall, items_count_in_class, total_items):
    f1 = []
    for i in range(k):
        f1.append(calc_f1(precision[i], recall[i]) * items_count_in_class[i])
    weighted_f1 = sum(f1) / total_items
    return weighted_f1


def calc_macro_f1(k, precision, recall, items_count_in_class, total_items):
    precision_w = sum([precision[i] * items_count_in_class[i] for i in range(k)]) / total_items
    recall_w = sum([recall[i] * items_count_in_class[i] for i in range(k)]) / total_items
    macro_f1 = calc_f1(precision_w, recall_w)
    return macro_f1


def calc_f1(precision, recall):
    if precision + recall != 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


def main():
    k = int(input())
    confusion_matrix = [[int(x) for x in input().split()] for _ in range(k)]
    for measure in calc_measures(k, confusion_matrix):
        print(measure)


if __name__ == '__main__':
    main()
