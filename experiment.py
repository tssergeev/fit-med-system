import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import random

# ========== НАСТРОЙКИ ==========
np.random.seed(42)
random.seed(42)
OUT_DIR = "experiment_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ ==========
n_users = 500

# ---- Источник А: реляционная БД фитнес-клуба (SQL-подобный) ----
users = []
for i in range(n_users):
    age = np.random.randint(18, 65)
    gender = np.random.choice(['M', 'F'], p=[0.6, 0.4])
    weight = round(np.random.normal(70 if gender=='M' else 60, 12), 1)
    height = round(np.random.normal(175 if gender=='M' else 163, 8), 1)
    club_visits = np.random.poisson(lam=3)  # средняя частота посещений в неделю
    users.append([i+1, age, gender, weight, height, club_visits])

df_sql = pd.DataFrame(users, columns=['user_id', 'age', 'gender', 'weight_kg', 'height_cm', 'weekly_visits'])
df_sql.to_csv(os.path.join(OUT_DIR, 'table1_club_members.csv'), index=False)

# ---- Источник Б: API фитнес-трекера (данные о пульсе, HRV, сне) ----
tracker = []
for i in range(n_users):
    rest_hr = np.random.randint(50, 90)
    hrv = round(np.random.normal(45, 15), 1)
    hrv = max(20, min(80, hrv))
    sleep_hours = round(np.random.normal(7, 1.2), 1)
    sleep_hours = max(4, min(10, sleep_hours))
    tracker.append([i+1, rest_hr, hrv, sleep_hours])

df_api = pd.DataFrame(tracker, columns=['user_id', 'resting_hr', 'hrv_ms', 'sleep_hours'])
df_api.to_csv(os.path.join(OUT_DIR, 'table2_health_metrics.csv'), index=False)

# ---- Источник В: файловое хранилище (CSV) с активностями ----
activities = []
for idx, row in df_sql.iterrows():
    user_id = row['user_id']
    weekly = row['weekly_visits']
    # За 90 дней (около 12 недель) ожидаемое число активностей = weekly * 12
    # Добавим случайный разброс
    expected = int(weekly * 12)
    num_acts = np.random.poisson(lam=expected)
    num_acts = min(num_acts, 100)  # ограничим сверху
    for _ in range(num_acts):
        date = datetime.now() - timedelta(days=np.random.randint(0, 90))
        act_type = np.random.choice(['running', 'cycling', 'swimming', 'strength'], p=[0.4,0.2,0.1,0.3])
        duration = np.random.randint(20, 120)
        intensity = np.random.choice(['low', 'medium', 'high'], p=[0.2,0.5,0.3])
        activities.append([user_id, date.strftime('%Y-%m-%d'), act_type, duration, intensity])

df_activities = pd.DataFrame(activities, columns=['user_id', 'date', 'activity_type', 'duration_min', 'intensity'])
df_activities.to_csv(os.path.join(OUT_DIR, 'table3_activities.csv'), index=False)

# Статус открытости (для поиска)
openness = []
for i in range(n_users):
    # 30% пользователей открыты к предложениям
    is_open = np.random.choice([0,1], p=[0.7,0.3])
    # некоторые открыты только частично
    data_categories = ['basic', 'health', 'activities']
    visible = np.random.choice(data_categories, size=np.random.randint(1,4), replace=False).tolist() if is_open else []
    openness.append([i+1, is_open, ','.join(visible)])

df_open = pd.DataFrame(openness, columns=['user_id', 'open_to_offers', 'visible_data'])
df_open.to_csv(os.path.join(OUT_DIR, 'user_openness.csv'), index=False)

print("Сгенерировано 3 таблицы с данными (сохранены в папку experiment_results)")

# ========== 2. МЕДИАТОР: объединение данных ==========
# Глобальная виртуальная схема – объединение всех источников
df_merged = df_sql.merge(df_api, on='user_id', how='left')
df_merged = df_merged.merge(df_open, on='user_id', how='left')
# Агрегируем активности: средняя продолжительность, кол-во, последняя активность
act_agg = df_activities.groupby('user_id').agg(
    total_activities=('duration_min', 'count'),
    avg_duration=('duration_min', 'mean'),
    last_activity=('date', 'max')
).reset_index()
df_merged = df_merged.merge(act_agg, on='user_id', how='left')
df_merged.fillna({'total_activities':0, 'avg_duration':0, 'last_activity':'2020-01-01'}, inplace=True)

# ========== 3. ПОИСК ПО БИОМЕТРИЧЕСКИМ КРИТЕРИЯМ ==========
# Пример запроса тренера: женщины 25-35 лет, HRV > 50, weekly_visits >= 2, open_to_offers=1
query_results = df_merged[
    (df_merged['gender'] == 'F') &
    (df_merged['age'].between(25,35)) &
    (df_merged['hrv_ms'] > 50) &
    (df_merged['weekly_visits'] >= 2) &
    (df_merged['open_to_offers'] == 1)
].copy()

print(f"\nНайдено пользователей по запросу тренера: {len(query_results)}")
if len(query_results) > 0:
    print(query_results[['user_id','age','hrv_ms','weekly_visits','total_activities']].head(10))

# ========== 4. ПОСТРОЕНИЕ ГРАФИКОВ (4 шт) ==========
sns.set_style("whitegrid")

# График 1: распределение HRV по статусу открытости
plt.figure(figsize=(8,5))
sns.histplot(data=df_merged, x='hrv_ms', hue='open_to_offers', bins=20, kde=True, palette='Set2')
plt.title('Распределение вариабельности сердечного ритма (HRV)\nв зависимости от статуса открытости')
plt.xlabel('HRV, мс')
plt.ylabel('Количество пользователей')
plt.legend(title='Открыт к предложениям', labels=['Нет', 'Да'])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_hrv_distribution.png'), dpi=150)
plt.close()

# График 2: средняя частота посещений в неделю по возрастным группам
df_merged['age_group'] = pd.cut(df_merged['age'], bins=[18,30,40,50,65], labels=['18-30','31-40','41-50','51-65'])
plt.figure(figsize=(8,5))
sns.boxplot(data=df_merged, x='age_group', y='weekly_visits', hue='open_to_offers', palette='Set1')
plt.title('Еженедельные посещения клуба по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Посещений в неделю')
plt.legend(title='Открыт к предложениям')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_visits_by_age.png'), dpi=150)
plt.close()

# График 3: корреляция между HRV и длительностью сна
plt.figure(figsize=(8,5))
sns.scatterplot(data=df_merged, x='sleep_hours', y='hrv_ms', hue='open_to_offers', alpha=0.7, palette='viridis')
plt.title('Взаимосвязь между длительностью сна и HRV')
plt.xlabel('Сон, часы')
plt.ylabel('HRV, мс')
plt.legend(title='Открыт к предложениям')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_sleep_hrv_corr.png'), dpi=150)
plt.close()

# График 4: Топ-5 наиболее активных пользователей (исправленный)
act_count = df_activities.groupby('user_id').size().reset_index(name='total_activities')
top5 = act_count.nlargest(5, 'total_activities')
plt.figure(figsize=(8,5))
sns.barplot(data=top5, x='user_id', y='total_activities', palette='Blues_d')
plt.title('Топ-5 пользователей по количеству тренировок за 90 дней')
plt.xlabel('ID пользователя')
plt.ylabel('Количество активностей')
for i, val in enumerate(top5['total_activities']):
    plt.text(i, val + 0.2, str(val), ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_top_active_users.png'), dpi=150)
plt.close()

print("Построено 4 графика (сохранены в experiment_results)")
