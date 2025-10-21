import streamlit as st
import pandas as pd
import requests
import io

def calculate_relatives(row):
    return row['SibSp'] + row['Parch']


def load_data():
    url = "https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df

def main():
    st.title("Анализ пассажиров Титаника")

    st.image("960.jpg",
             caption="Титаник", use_column_width=True)

    df = load_data()

    sex = st.selectbox("Выберите пол пассажиров:", ["Все", "Мужчины", "Женщины"])

    if sex == "Мужчины":
        filtered_df = df[df['Sex'] == 'male']
    elif sex == "Женщины":
        filtered_df = df[df['Sex'] == 'female']
    else:
        filtered_df = df

    survived_data = filtered_df[filtered_df['Survived'] == 1]
    dead_data = filtered_df[filtered_df['Survived'] == 0]

    if len(survived_data) > 0:
        survived_relatives_mean = survived_data.apply(calculate_relatives, axis=1).mean()
    else:
        survived_relatives_mean = 0

    if len(dead_data) > 0:
        dead_relatives_mean = dead_data.apply(calculate_relatives, axis=1).mean()
    else:
        dead_relatives_mean = 0

    results_df = pd.DataFrame({
        'Категория': ['Выжившие', 'Погибшие'],
        'Количество пассажиров': [len(survived_data), len(dead_data)],
        'Среднее число родственников': [
            round(survived_relatives_mean, 3),
            round(dead_relatives_mean, 3)
        ]
    })

    st.table(results_df)

if __name__ == "__main__":
    main()