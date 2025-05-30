import os.path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

TRUE_PIXELS_PATH = 'data/true_pixels.csv'
FALSE_PIXELS_PATH = 'data/false_pixels.csv'
MODEL_NAME = 'model_random_forest_full'
MODELS_PATH = 'models'


def main():
    # Загрузка и подготовка датасета
    true_pixels = pd.read_csv(TRUE_PIXELS_PATH, index_col=0)
    false_pixels = pd.read_csv(FALSE_PIXELS_PATH, index_col=0)
    df = pd.concat([true_pixels, false_pixels])
    df = df.sample(frac=1)

    df['delta_red'] = df['red'] - df['prev_red']
    df['delta_green'] = df['green'] - df['prev_green']
    df['delta_blue'] = df['blue'] - df['prev_blue']

    df_full = df.copy()
    df_may = df[df['month'] == 'may'].copy()
    df_june = df[df['month'] == 'june'].copy()
    df_july = df[df['month'] == 'july'].copy()
    df_august = df[df['month'] == 'august'].copy()
    df_september = df[df['month'] == 'september'].copy()

    months_df = [df_may, df_june, df_july, df_august, df_september]

    # Признаки для обучения
    features = ['red', 'green', 'blue', 'delta_red', 'delta_green', 'delta_blue', 'prev_ndvi', 'delta_ndvi',
                'mean_delta_ndvi', 'prev_evi', 'delta_evi', 'mean_delta_evi']

    # Подготовка датасета для обучения
    x_train, x_val, y_train, y_val = train_test_split(df_full, df_full['forest_change'], test_size=0.15,
                                                      random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_train, x_train['forest_change'], test_size=0.15,
                                                        random_state=42)

    # Обучение и сохранение модели
    model_random_forest_full = RandomForestClassifier(n_estimators=30, max_depth=15, random_state=42)
    model_random_forest_full.fit(x_train[features], y_train)
    with open(os.path.join(MODELS_PATH, f'{MODEL_NAME}.pkl'), 'wb') as fp:
        pickle.dump(model_random_forest_full, fp)

    # Сбор метрик по месяцам
    month_accuracies = []
    month_precisions = []
    month_recalls = []
    month_f1_scores = []

    for month_df in months_df:
        x_test_df = month_df[features]
        y_test_df = month_df['forest_change']
        y_pred = model_random_forest_full.predict(x_test_df)

        month_accuracies.append(round(accuracy_score(y_test_df, y_pred), 4))
        month_precisions.append(round(precision_score(y_test_df, y_pred, average='weighted'), 4))
        month_recalls.append(round(recall_score(y_test_df, y_pred, average='weighted'), 4))
        month_f1_scores.append(round(f1_score(y_test_df, y_pred, average='weighted'), 4))

    # Сбор метрик для всего датасета
    full_accuracies = []
    full_precisions = []
    full_recalls = []
    full_f1_scores = []

    x_test_df = x_test[features]
    y_test_df = x_test['forest_change']
    y_pred = model_random_forest_full.predict(x_test_df)

    full_accuracies.append(round(accuracy_score(y_test_df, y_pred), 4))
    full_precisions.append(round(precision_score(y_test_df, y_pred, average='weighted'), 4))
    full_recalls.append(round(recall_score(y_test_df, y_pred, average='weighted'), 4))
    full_f1_scores.append(round(f1_score(y_test_df, y_pred, average='weighted'), 4))

    # Оценка качества работы модели по месяцам
    print(f'Оценка качества работы модели {MODEL_NAME} по месяцам')
    month_names = ['may', 'june', 'july', 'august', 'september']
    results = {
        'Month': month_names,
        'Accuracy': month_accuracies,
        'Precision': month_precisions,
        'Recall': month_recalls,
        'F1-Score': month_f1_scores,
    }
    result_df = pd.DataFrame(results)
    print(result_df)
    print()

    # Оценка качества работы модели на всём датасете
    print(f'Оценка качества работы модели {MODEL_NAME} на всём датасете')
    model_names = [MODEL_NAME]
    results = {
        'Model': model_names,
        'Accuracy': full_accuracies,
        'Precision': full_precisions,
        'Recall': full_recalls,
        'F1-Score': full_f1_scores,
    }
    result_df = pd.DataFrame(results)
    print(result_df)


if __name__ == '__main__':
    main()