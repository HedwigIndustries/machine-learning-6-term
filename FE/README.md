# Лабораторная работа №5: **Выбор признаков**

## Задание

### Реализуйте три метода выбора признаков:
1. **Встроенный метод (Embedded)** — метод, который строится в процессе обучения модели.
2. **Метод обёртки (Wrapper)** — метод, при котором выбор признаков оценивается по качеству модели.
3. **Фильтрующий метод (Filter)** — метод, который оценивает значимость признаков, не привязываясь к модели.

### Применение методов:
1. Примените реализованные методы на наборе данных:
   - **SMS Spam Collection** (доступен на [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)).
   - **Castle-or-lock** (или другой набор данных по вашему выбору).

2. **Выведите 30 наиболее значимых признаков (слов)**, выбранных каждым методом.

3. **Сравните списки признаков**, выбранных реализованными методами, с результатами трёх библиотечных методов:
   - Эти методы могут быть любыми и необязательно такими же, как реализованные (можно использовать все фильтрующие или любые комбинации).
   - Примеры библиотечных методов: [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html), [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html), [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

---

## Оценка качества моделей

1. Выберите **не менее трёх классификаторов**, например:
   - **Logistic Regression** ([LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)).
   - **Random Forest** ([RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)).
   - **Support Vector Machine** ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)).

2. **Оцените качество работы каждого классификатора**:
   - До выбора признаков.
   - После выбора признаков для каждого метода.

### Метрики оценки:
- Используйте метрики, такие как **accuracy**, **precision**, **recall**, или **F1-score**, чтобы сравнить, как изменяется качество моделей до и после выбора признаков.

---

## Дополнительные материалы

- Примеры реализации методов выбора признаков:
   - [Embedded methods](https://scikit-learn.org/stable/modules/feature_selection.html#embedded-methods).
   - [Filter methods](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).
   - [Wrapper methods](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination).
