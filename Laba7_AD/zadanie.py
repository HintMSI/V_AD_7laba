import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# 1. Загрузка данных
df = pd.read_csv('baza.csv', sep=';', encoding='cp1251')

print("Размер исходного датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

# 2. Предварительная фильтрация
# a.
df = df[df['ВидПомещения'] == 'жилые помещения']
print(f"\nРазмер после фильтрации по виду помещения: {df.shape}")

# b.
df = df[df['СледующийСтатус'].isin(['Продана', 'Свободна'])] # isin проверяет есть ли каждое значение столбца в заданном списке
df['СледующийСтатус'] = df['СледующийСтатус'].map({'Продана': 1, 'Свободна': 0}) # map заменяет значения в столбцах согласно словарю,тоесть продана на 1;Свободна на 0.
print(f"Размер после фильтрации по статусу: {df.shape}")

# c.
columns_to_drop = ['УИД_Брони', 'ВидПомещения','ДатаБрони','ВремяБрони']
df = df.drop(columns=columns_to_drop, errors='ignore')

# 3. Проверка типов данных и преобразование к числовому типу
print("\nТипы данных перед преобразованием:")
print(df.dtypes)

# a. Проверка числовых полей
numeric_columns = ['ПродаваемаяПлощадь', 'Этаж', 'СтоимостьНаДатуБрони',
                   'СкидкаНаКвартиру', 'ФактическаяСтоимостьПомещения']
#errors='coerce'- если встречается значение, которое нельзя преобразовать в число, оно заменяется на NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# b. Кодирование бинарных признаков
binary_mappings = {
    'ИсточникБрони': {'ручная': 0, 'МП': 1},
    'ВременнаяБронь': {'Нет': 0, 'Да': 1},
    'ТипСтоимости': {'Стоимость при 100% оплате': 0, 'Стоимость в рассрочку': 1},
    'ВариантОплаты': {'Единовременная оплата': 0, 'Оплата в рассрочку': 1},
    'ВариантОплатыДоп': {'Ипотека': 0, 'Вторичное жилье': 1},
    'СделкаАН': {'Нет': 0, 'Да': 1},
    'ИнвестиционныйПродукт': {'Нет': 0, 'Да': 1},
    'Привилегия': {'Нет': 0, 'Да': 1}
}

# возвращает пары (ключ, значение) из словаря
for col, mapping in binary_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# c. One-hot кодирование для категориальных признаков
categorical_columns = ['Город', 'Статус лида (из CRM)']
for col in categorical_columns:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=[col]) #prefix=[col] - префикс для новых столбцов ('Город_...')


# d. Преобразование поля "Тип"
def convert_type(value):
    if pd.isna(value): # если уже значение пустое,то просто возвращает Nan
        return np.nan
    if isinstance(value, str): #проверка ,является ли значние - стркоой
        value_str = str(value).strip() #удаление пробелов
        if value_str.endswith('к'):
            return float(value_str.replace('к', '').replace(',', '.'))
        elif value_str == 'с':
            return np.nan
        elif value_str.replace('.', '').replace(',', '').isdigit():
            return float(value_str.replace(',', '.')) # заменяет . на , и проверяет на число , и если так,то заменяет на правильное число типа float
    return float(value) if pd.notna(value) else np.nan


df['Тип'] = df['Тип'].apply(convert_type)
# Убедимся, что все данные числовые
print("\nТипы данных после преобразования:")
print(df.dtypes)

# 4. Обработка отсутствующих данных
print("\nПропущенные значения до обработки:")
print(df.isnull().sum())

# a. СкидкаНаКвартиру - заменяем на 0
if 'СкидкаНаКвартиру' in df.columns:
    df['СкидкаНаКвартиру'] = df['СкидкаНаКвартиру'].fillna(0)

# b. Тип и ПродаваемаяПлощадь - заменяем на медиану
if 'Тип' in df.columns:
    median_type = df['Тип'].median()
    df['Тип'] = df['Тип'].fillna(median_type)

if 'ПродаваемаяПлощадь' in df.columns:
    median_area = df['ПродаваемаяПлощадь'].median()
    df['ПродаваемаяПлощадь'] = df['ПродаваемаяПлощадь'].fillna(median_area)

# c. ВариантОплатыДоп - удаляем столбец
df = df.drop('ВариантОплатыДоп', axis=1, errors='ignore') #axis=1 ,тк мы удаляем столбец,а строку

# d. Удаляем строки с оставшимися пропущенными значениями
df = df.dropna()
print(f"\nРазмер после обработки пропусков: {df.shape}")

# 5. Дополнение данных
# a. Цена за квадратный метр
df['Цена_за_кв_метр'] = df['ФактическаяСтоимостьПомещения'] / df['ПродаваемаяПлощадь']

# b. Скидка в процентах
df['Скидка_в_процентах'] = (df['СкидкаНаКвартиру'] / df['ФактическаяСтоимостьПомещения']) * 100

print("\nНовые признаки:")
print(df[['Цена_за_кв_метр', 'Скидка_в_процентах']].head())

# 6. Нормализация
# Целевой признак - СледующийСтатус,тк в задании "Для планирования своих финансовых показателей компания хочет прогнозировать, приведет ли бронирование к заключению договора или нет."
# если пусто или «В резерве» — статус еще не определен, «Продана» — договор оформлен, «Свободна» — бронь снят.
# Поэтому и выбираем его как целевой,который нужно научиться предскахзывать.
target_column = 'СледующийСтатус'
y = df[target_column].copy()
# Исключаем целевую переменную
exclude_from_normalization = [target_column]

# Все числовые колонки кроме целевой
numeric_columns_for_normalization = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns_for_normalization = [col for col in numeric_columns_for_normalization
                                     if col not in exclude_from_normalization]

# Создаем копию df для нормализованных данных
df_normalized = df.copy()

# Берем из библиотеки scikit объект MinMaxScaler,который будет преобразовывать значения признаков
# к диапазону [0,1] по формуле X_normalized = (X - X_min) / (X_max - X_min)
scaler = MinMaxScaler()
df_normalized[numeric_columns_for_normalization] = scaler.fit_transform(
    df[numeric_columns_for_normalization]
)
#создается уже нормализованный ДФ,в котором функция fit() - вычисляет min,max для столбцов
#transform()-уже применяет нормализацию к данным

# Особенная нормализация для СкидкаНаКвартиру
if 'СкидкаНаКвартиру' in df_normalized.columns:
    df_normalized['СкидкаНаКвартиру'] = df_normalized['СкидкаНаКвартиру'] - 0.5
# тоесть тут по сути обычная нормализация,однако когда мы вычитаем 0.5,то это сдвигает диапазон к [-0.5;0.5]

# Восстанавливаем целевую переменную
df_normalized[target_column] = y

# 7. Проверка сбалансированности датасета
target_counts = df_normalized['СледующийСтатус'].value_counts()
#Это считает, сколько строк принадлежит к каждому классу:
#Класс 0 ("Свободна") - сколько бронирований не привели к продаже
#Класс 1 ("Продана") - сколько бронирований привели к продаже
print(f"\nРаспределение целевого признака:")
print(target_counts)
print(f"Соотношение: {target_counts[1] / target_counts[0]:.2f}")

if min(target_counts) / max(target_counts) > 0.5:
    print("Датасет считается сбалансированным")
else:
    print("Датасет НЕ сбалансирован")

# 8. Формирование списка признаков

# 8.1. Расчет корреляционной матрицы
correlation_matrix = df.corr(numeric_only=True)
print("\nКорреляционная матрица с целевым признаком:")
print(correlation_matrix['СледующийСтатус'].sort_values(ascending=False))

# 8.2. Выбор наиболее коррелирующих признаков с целевой переменной
target_correlations = correlation_matrix['СледующийСтатус'].abs().sort_values(ascending=False)
# Исключаем сам целевой признак и ьерем тиоп 5 признаков
selected_features = target_correlations[1:6].index.tolist()

print(f"\nВыбранные признаки: {selected_features}")

target_column ='СледующийСтатус' # целевой признак

# 9. Разбиение на обучающую и тестовую выборки
X = df_normalized[selected_features]
y = df_normalized[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64, stratify=y)
# test_size=0.3 : 30% данных идет в тестовую выборку, 70% - в обучающую
# random_state=64  при каждом запуске кода будет одно и то же разбиение, но данные все равно перемешиваются в случайном порядке перед разбиением
# stratify=y - доп стратификация ( для сохранение пропорций классов)
print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка: {X_test.shape[0]} строк")

# 10. Модель KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# 11. Модель Decision Tree
tree_model = DecisionTreeClassifier(random_state=64)
tree_model.fit(X_train, y_train)

print("\nМодели обучены успешно")

# 12. Получение прогнозов
# KNN
knn_train_pred = knn_model.predict(X_train)
knn_test_pred = knn_model.predict(X_test)

# Decision Tree
tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)


# 13. Расчет показателей качества
def calculate_metrics(y_true, y_pred, model_name, dataset_type):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"{model_name} ({dataset_type}):")
    print(f"  F-мера: {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print()

    return f1, precision, recall


print("ПОКАЗАТЕЛИ КАЧЕСТВА:")
print("=" * 20)

# KNN
knn_train_metrics = calculate_metrics(y_train, knn_train_pred, "KNN", "Обучающая")
knn_test_metrics = calculate_metrics(y_test, knn_test_pred, "KNN", "Тестовая")

# Decision Tree
tree_train_metrics = calculate_metrics(y_train, tree_train_pred, "Decision Tree", "Обучающая")
tree_test_metrics = calculate_metrics(y_test, tree_test_pred, "Decision Tree", "Тестовая")

# Вывод:
#Лучшая модель оказалась KNN,тк на тестовой выборке у нее больше всего F-мера
# и разница между обучающей и тестовой выборкой всего 0.008,что очень мало
#Для Decision Tree все тоже неплохо,однако уступает по F-мере и разница в выборках 0.04
#
#KNN:
#1.Precision = 0.896 - это означает, что когда модель говорит "квартира будет продана", она права в 89.6% случаев
#2.Recall = 0.772 - означает, что модель находит 77.2% всех фактически проданных квартир
#3.Высокий Precision, умеренный Recall - модель редко ошибается, когда предсказывает продажу, но пропускает около 22,8% реальных продаж
#Decision tree:
#1.Precision = 0.885 - права в 88.5% случаев
#2.Recall = 0.74 -находит только 74% реальных продаж

