import pytest
import pandas as pd
from Create import calculate_relatives

# 1. Проверка корректности подсчета родственников
def test_calculate_relatives():
    row = pd.Series({'SibSp': 2, 'Parch': 1})
    result = calculate_relatives(row)
    assert result == 3, f"Expected 3, but got {result}"

# 2. Подсчет среднего числа родственников среди выживших мужчин
def test_survived_men():
    data = pd.DataFrame({
        'Sex': ['male', 'male', 'female', 'male'],
        'Survived': [1, 1, 0, 1],
        'SibSp': [1, 0, 2, 1],
        'Parch': [0, 1, 0, 1]
    })
    survived_data = data[(data['Survived'] == 1) & (data['Sex'] == 'male')]
    mean_relatives = survived_data.apply(calculate_relatives, axis=1).mean()
    assert round(mean_relatives, 3) == 1.0, f"Expected 1.0, but got {mean_relatives}"

# 3. Подсчет среднего числа родственников среди погибших женщин
def test_dead_women():
    data = pd.DataFrame({
        'Sex': ['male', 'male', 'female', 'female'],
        'Survived': [0, 1, 0, 0],
        'SibSp': [1, 0, 2, 1],
        'Parch': [0, 1, 0, 1]
    })
    dead_data = data[(data['Survived'] == 0) & (data['Sex'] == 'female')]
    mean_relatives = dead_data.apply(calculate_relatives, axis=1).mean()
    assert round(mean_relatives, 3) == 2.0, f"Expected 2.0, but got {mean_relatives}"

# 4. Подсчет среднего числа родственников среди всех пассажиров (без фильтрации по полу)
def test_all_passengers():
    data = pd.DataFrame({
        'Sex': ['male', 'female', 'female', 'male'],
        'Survived': [1, 0, 1, 0],
        'SibSp': [1, 2, 3, 0],
        'Parch': [0, 1, 1, 1]
    })
    all_data = data
    survived_data = all_data[all_data['Survived'] == 1]
    dead_data = all_data[all_data['Survived'] == 0]

    survived_relatives_mean = survived_data.apply(calculate_relatives, axis=1).mean() if len(survived_data) > 0 else 0
    dead_relatives_mean = dead_data.apply(calculate_relatives, axis=1).mean() if len(dead_data) > 0 else 0

    assert round(survived_relatives_mean, 3) == 2.0, f"Expected 2.0 for survived, but got {survived_relatives_mean}"
    assert round(dead_relatives_mean, 3) == 1.333, f"Expected 1.333 for dead, but got {dead_relatives_mean}"


if __name__ == "__main__":
    pytest.main()
