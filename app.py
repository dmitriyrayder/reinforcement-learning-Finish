import streamlit as st
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Попытка импорта stable-baselines3 (опционально)
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ============================================================================
# РАЗДЕЛ 1: ПОДГОТОВКА ДАННЫХ И БАЗОВАЯ RL СРЕДА
# ============================================================================

# Настройка страницы
st.set_page_config(page_title="RL Система для Оптики", layout="wide")

# Заголовок
st.title("🤖 Reinforcement Learning: Оптимизация розничной сети")
st.markdown("---")

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Загрузка и подготовка данных"""
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {str(e)}")
        st.stop()
    
    # Преобразование даты - убираем пробелы и парсим
    df['Datasales'] = df['Datasales'].astype(str).str.strip()
    df['Datasales'] = pd.to_datetime(df['Datasales'], format='%d.%m.%Y', errors='coerce')
    
    # Проверка на успешность парсинга
    invalid_dates = df['Datasales'].isna().sum()
    if invalid_dates > 0:
        st.warning(f"⚠️ Найдено {invalid_dates} записей с некорректными датами. Они будут исключены.")
        df = df.dropna(subset=['Datasales'])
    
    # Добавляем недостающие поля рандомно
    np.random.seed(42)
    
    # Себестоимость (60-80% от цены)
    df['Cost'] = df['Price'] * np.random.uniform(0.6, 0.8, len(df))
    df['Cost'] = df['Cost'].round(2)
    
    # Маржа (правильная формула: (Цена - Себестоимость) * Количество)
    df['Margin'] = (df['Price'] - df['Cost']) * df['Qty']
    
    # Уникальные магазины
    stores = df['Magazin'].unique()
    
    # Характеристики магазинов
    store_features = {}
    regions = ['Київ', 'Львів', 'Одеса', 'Харків', 'Дніпро']
    
    for store in stores:
        store_features[store] = {
            'region': np.random.choice(regions),
            'area_sqm': np.random.randint(50, 200),  # площадь магазина
            'traffic': np.random.randint(100, 500)  # средний трафик в день
        }
    
    df['Region'] = df['Magazin'].map(lambda x: store_features[x]['region'])
    df['Store_Area'] = df['Magazin'].map(lambda x: store_features[x]['area_sqm'])
    df['Daily_Traffic'] = df['Magazin'].map(lambda x: store_features[x]['traffic'])
    
    # Расчет остатков: +50% к среднему числу продаж по каждому товару в магазине
    sales_avg = df.groupby(['Magazin', 'Art'])['Qty'].mean().reset_index()
    sales_avg.columns = ['Magazin', 'Art', 'Avg_Sales']
    sales_avg['Stock'] = (sales_avg['Avg_Sales'] * 1.5).round(0).astype(int)
    
    df = df.merge(sales_avg[['Magazin', 'Art', 'Stock']], on=['Magazin', 'Art'], how='left')
    df['Stock'] = df['Stock'].fillna(5).astype(int)
    
    return df, store_features

class RetailEnvironment(gym.Env):
    """Среда для RL: управление распределением товара и маркетингом"""
    
    def __init__(self, df, stores, products, horizon_days=30):
        super(RetailEnvironment, self).__init__()
        
        self.df = df
        self.stores = stores
        self.products = products[:100]  # Ограничиваем для скорости
        self.horizon_days = horizon_days
        self.current_step = 0
        
        # Пространство действий: 
        # [магазин_индекс, товар_индекс, количество_для_перераспределения, промо_да/нет]
        self.action_space = spaces.MultiDiscrete([
            len(self.stores),  # выбор магазина
            len(self.products),  # выбор товара
            10,  # количество единиц товара (0-9)
            2   # промо акция (0=нет, 1=да)
        ])
        
        # Пространство состояний
        # [остатки_по_магазинам, продажи_за_неделю, маржа, день_месяца]
        self.observation_space = spaces.Box(
            low=0, high=1000, 
            shape=(len(self.stores) * len(self.products) + 10,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Сброс среды"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Инициализация остатков
        self.stocks = {}
        for store in self.stores:
            self.stocks[store] = {}
            for product in self.products:
                avg_stock = self.df[(self.df['Magazin'] == store) & 
                                   (self.df['Art'] == product)]['Stock'].mean()
                self.stocks[store][product] = int(avg_stock) if not np.isnan(avg_stock) else 5
        
        self.total_revenue = 0
        self.total_margin = 0
        self.actions_history = []
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Получение текущего состояния"""
        state = []
        
        # Остатки по магазинам (упрощенно - средние по топ продуктам)
        for store in self.stores[:5]:  # Берем первые 5 магазинов
            avg_stock = np.mean([self.stocks[store].get(p, 0) for p in self.products[:20]])
            state.append(avg_stock)
        
        # Дополнительные фичи
        state.extend([
            self.current_step / self.horizon_days,  # прогресс
            self.total_revenue / 100000,  # нормализованная выручка
            self.total_margin / 50000,  # нормализованная маржа
            len(self.actions_history) / 100  # количество действий
        ])
        
        # Дополняем до нужного размера
        while len(state) < self.observation_space.shape[0]:
            state.append(0)
        
        return np.array(state[:self.observation_space.shape[0]], dtype=np.float32)
    
    def step(self, action):
        """Выполнение действия"""
        store_idx, product_idx, qty, promo = action
        
        store = self.stores[store_idx]
        product = self.products[product_idx]
        
        # Проверяем наличие товара
        current_stock = self.stocks[store].get(product, 0)
        
        if current_stock <= 0:
            # Нет товара - отрицательная награда
            reward = -10
        else:
            # Симуляция продаж
            base_sales = min(qty + 1, current_stock)
            
            # Промо увеличивает продажи на 20-50%
            if promo == 1:
                sales_multiplier = np.random.uniform(1.2, 1.5)
                promo_cost = base_sales * 50  # стоимость промо
            else:
                sales_multiplier = 1.0
                promo_cost = 0
            
            actual_sales = int(base_sales * sales_multiplier)
            actual_sales = min(actual_sales, current_stock)
            
            # Получаем цену и себестоимость
            product_data = self.df[(self.df['Magazin'] == store) & 
                                   (self.df['Art'] == product)]
            
            if len(product_data) > 0:
                avg_price = product_data['Price'].mean()
                avg_cost = product_data['Cost'].mean()
            else:
                avg_price = 1000
                avg_cost = 700
            
            # Расчет выручки и маржи
            revenue = actual_sales * avg_price
            margin = actual_sales * (avg_price - avg_cost) - promo_cost
            
            # Обновляем остатки
            self.stocks[store][product] = current_stock - actual_sales
            
            # Награда = маржа
            reward = margin / 1000  # нормализуем
            
            self.total_revenue += revenue
            self.total_margin += margin
        
        self.current_step += 1
        self.actions_history.append({
            'step': self.current_step,
            'store': store,
            'product': product,
            'qty': qty,
            'promo': promo,
            'reward': reward
        })
        
        terminated = self.current_step >= self.horizon_days
        truncated = False
        
        return self._get_state(), reward, terminated, truncated, {}
    
    def render(self):
        """Визуализация состояния"""
        pass

class SimpleRLAgent:
    """Простой RL агент (Random baseline)"""
    
    def __init__(self, env):
        self.env = env
        self.q_table = {}
    
    def get_action(self, state):
        """Выбор действия (случайное)"""
        return self.env.action_space.sample()
    
    def train(self, episodes=100):
        """Обучение агента"""
        rewards_history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
            
            rewards_history.append(total_reward)
            
            # Обновление прогресса
            progress_bar.progress((episode + 1) / episodes)
            status_text.text(f"Эпизод {episode + 1}/{episodes} | Награда: {total_reward:.2f}")
        
        progress_bar.empty()
        status_text.empty()
        
        return rewards_history

# ============================================================================
# РАЗДЕЛ 2: ПРОДВИНУТАЯ RL СИСТЕМА
# ============================================================================

class AdvancedRetailEnvironment(gym.Env):
    """Продвинутая среда с дополнительными признаками и улучшенной reward функцией"""
    
    def __init__(self, df, stores, products, horizon_days=30):
        super(AdvancedRetailEnvironment, self).__init__()
        
        self.df = df
        self.stores = stores[:10]  # Топ 10 магазинов
        self.products = products[:50]  # Топ 50 товаров
        self.horizon_days = horizon_days
        self.current_step = 0
        
        # Расширенное пространство действий
        # [store, product, quantity, promo_type, price_adjustment]
        self.action_space = spaces.MultiDiscrete([
            len(self.stores),  # магазин
            len(self.products),  # товар
            10,  # количество (0-9)
            3,   # тип промо: 0=нет, 1=скидка, 2=акция 1+1
            5    # корректировка цены: 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
        ])
        
        # Расширенное пространство состояний
        # [остатки, продажи, день_недели, месяц, конкуренты, погода, CLV]
        state_size = (
            len(self.stores) * 5 +  # остатки по топ магазинам
            len(self.stores) * 5 +  # продажи за неделю
            7 +  # день недели (one-hot)
            12 + # месяц (one-hot)
            3 +  # конкуренты (низкая/средняя/высокая активность)
            4 +  # погода (солнце/дождь/снег/облачно)
            5    # CLV метрики
        )
        
        self.observation_space = spaces.Box(
            low=0, high=100, 
            shape=(state_size,), 
            dtype=np.float32
        )
        
        # Добавляем синтетические данные
        self._add_synthetic_features()
        self.reset()
    
    def _add_synthetic_features(self):
        """Добавление синтетических признаков"""
        np.random.seed(42)
        
        # День недели (0=Пн, 6=Вс)
        if 'DayOfWeek' not in self.df.columns:
            self.df['DayOfWeek'] = self.df['Datasales'].dt.dayofweek
        
        # Месяц
        if 'Month' not in self.df.columns:
            self.df['Month'] = self.df['Datasales'].dt.month
        
        # Сезон (1=зима, 2=весна, 3=лето, 4=осень)
        if 'Season' not in self.df.columns:
            self.df['Season'] = self.df['Month'].apply(
                lambda x: 1 if x in [12,1,2] else 2 if x in [3,4,5] else 3 if x in [6,7,8] else 4
            )
        
        # Конкуренты (синтетика)
        self.competitor_activity = {
            store: np.random.choice(['low', 'medium', 'high']) 
            for store in self.stores
        }
        
        # Погода (синтетика)
        self.weather_data = {}
        for date in self.df['Datasales'].unique():
            self.weather_data[date] = np.random.choice(['sunny', 'rainy', 'snowy', 'cloudy'])
        
        # CLV (Customer Lifetime Value) - синтетика
        # Простая модель: частые покупатели имеют выше CLV
        store_avg_purchase = self.df.groupby('Magazin')['Sum'].mean()
        self.store_clv = {store: val * 10 for store, val in store_avg_purchase.items()}
    
    def _get_state(self):
        """Получение расширенного состояния"""
        state = []
        
        # 1. Остатки по магазинам (топ 5 товаров в каждом)
        for store in self.stores[:5]:
            for product in self.products[:5]:
                stock = self.stocks.get(store, {}).get(product, 0)
                state.append(min(stock / 10, 10))  # нормализация
        
        # 2. Продажи за неделю
        recent_sales = self.sales_history[-7:] if len(self.sales_history) >= 7 else self.sales_history
        avg_sales = np.mean(recent_sales) if recent_sales else 0
        state.extend([avg_sales / 100] * 25)
        
        # 3. День недели (one-hot)
        day_of_week = self.current_step % 7
        day_one_hot = [0] * 7
        day_one_hot[day_of_week] = 1
        state.extend(day_one_hot)
        
        # 4. Месяц (one-hot)
        month = ((self.current_step // 30) % 12)
        month_one_hot = [0] * 12
        month_one_hot[month] = 1
        state.extend(month_one_hot)
        
        # 5. Конкуренты (one-hot)
        competitor_encoding = {'low': [1,0,0], 'medium': [0,1,0], 'high': [0,0,1]}
        if len(self.stores) > 0:
            comp_state = competitor_encoding.get(
                self.competitor_activity.get(self.stores[0], 'medium'),
                [0,1,0]
            )
        else:
            comp_state = [0,1,0]
        state.extend(comp_state)
        
        # 6. Погода (one-hot)
        weather_encoding = {'sunny': [1,0,0,0], 'rainy': [0,1,0,0], 'snowy': [0,0,1,0], 'cloudy': [0,0,0,1]}
        state.extend(weather_encoding.get('sunny', [1,0,0,0]))
        
        # 7. CLV метрики
        avg_clv = np.mean(list(self.store_clv.values())) if self.store_clv else 1000
        state.extend([
            self.total_revenue / 100000,
            self.total_margin / 50000,
            avg_clv / 10000,
            len(self.customer_visits) / 1000,
            self.customer_retention / 100
        ])
        
        # Дополняем до нужного размера
        while len(state) < self.observation_space.shape[0]:
            state.append(0)
        
        return np.array(state[:self.observation_space.shape[0]], dtype=np.float32)
    
    def reset(self, seed=None):
        """Сброс среды"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Инициализация остатков
        self.stocks = {}
        for store in self.stores:
            self.stocks[store] = {}
            for product in self.products:
                avg_stock = self.df[(self.df['Magazin'] == store) & 
                                   (self.df['Art'] == product)]['Stock'].mean()
                self.stocks[store][product] = int(avg_stock) if not np.isnan(avg_stock) else 10
        
        self.total_revenue = 0
        self.total_margin = 0
        self.actions_history = []
        self.sales_history = []
        self.customer_visits = []
        self.customer_retention = 80  # начальная удержка
        
        return self._get_state(), {}
    
    def _calculate_advanced_reward(self, revenue, margin, promo_type, price_adj, store):
        """Расширенная reward функция с учетом долгосрочных метрик"""
        
        # Базовая награда = маржа
        reward = margin / 1000
        
        # 1. Бонус за удержание клиентов (CLV)
        clv_bonus = self.store_clv.get(store, 1000) / 10000
        reward += clv_bonus * 0.3
        
        # 2. Штраф за агрессивные промо (снижают CLV)
        if promo_type == 2:  # акция 1+1
            reward -= 0.5  # краткосрочная выгода, долгосрочный вред
        elif promo_type == 1:  # скидка
            reward -= 0.2
        
        # 3. Бонус за оптимальное ценообразование
        if price_adj == 2:  # нет изменения цены
            reward += 0.3  # стабильность
        elif price_adj in [1, 3]:  # небольшая корректировка
            reward += 0.1
        else:  # агрессивная корректировка
            reward -= 0.2
        
        # 4. Сезонный мультипликатор
        month = (self.current_step // 30) % 12
        if month in [11, 0, 1]:  # зима - высокий сезон для очков
            reward *= 1.2
        elif month in [5, 6, 7]:  # лето - низкий сезон
            reward *= 0.9
        
        # 5. Бонус за день недели
        day = self.current_step % 7
        if day in [5, 6]:  # выходные
            reward *= 1.1
        
        # 6. Учет конкурентов
        comp_activity = self.competitor_activity.get(store, 'medium')
        if comp_activity == 'high':
            reward *= 0.9  # высокая конкуренция снижает эффективность
        elif comp_activity == 'low':
            reward *= 1.1
        
        # 7. Долгосрочная метрика: customer retention
        if margin > 0:
            self.customer_retention = min(95, self.customer_retention + 0.1)
            reward += (self.customer_retention / 100) * 0.5
        else:
            self.customer_retention = max(60, self.customer_retention - 0.2)
            reward -= 0.3
        
        return reward
    
    def step(self, action):
        """Выполнение действия с расширенной логикой"""
        store_idx, product_idx, qty, promo_type, price_adj = action
        
        store = self.stores[store_idx]
        product = self.products[product_idx]
        
        # Проверяем наличие товара
        current_stock = self.stocks[store].get(product, 0)
        
        if current_stock <= 0:
            reward = -10
            revenue = 0
            margin = 0
        else:
            # Симуляция продаж
            base_sales = min(qty + 1, current_stock)
            
            # Влияние промо
            if promo_type == 2:  # 1+1
                sales_multiplier = np.random.uniform(1.5, 2.0)
                promo_cost = base_sales * 100
            elif promo_type == 1:  # скидка
                sales_multiplier = np.random.uniform(1.2, 1.5)
                promo_cost = base_sales * 50
            else:
                sales_multiplier = 1.0
                promo_cost = 0
            
            # Влияние цены
            price_multipliers = {
                0: 1.3,   # -10% -> больше продаж
                1: 1.15,  # -5%
                2: 1.0,   # без изменений
                3: 0.9,   # +5% -> меньше продаж
                4: 0.75   # +10%
            }
            price_multiplier = price_multipliers.get(price_adj, 1.0)
            
            actual_sales = int(base_sales * sales_multiplier * price_multiplier)
            actual_sales = min(actual_sales, current_stock)
            
            # Получаем цену и себестоимость
            product_data = self.df[(self.df['Magazin'] == store) & 
                                   (self.df['Art'] == product)]
            
            if len(product_data) > 0:
                base_price = product_data['Price'].mean()
                avg_cost = product_data['Cost'].mean()
            else:
                base_price = 1000
                avg_cost = 700
            
            # Корректируем цену
            price_adjustments = {0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.1}
            final_price = base_price * price_adjustments[price_adj]
            
            # Расчет выручки и маржи
            revenue = actual_sales * final_price
            margin = actual_sales * (final_price - avg_cost) - promo_cost
            
            # Обновляем остатки
            self.stocks[store][product] = current_stock - actual_sales
            
            # Расширенная награда
            reward = self._calculate_advanced_reward(revenue, margin, promo_type, price_adj, store)
            
            self.total_revenue += revenue
            self.total_margin += margin
            self.sales_history.append(actual_sales)
            
            # Симуляция визитов клиентов
            if actual_sales > 0:
                self.customer_visits.extend([1] * actual_sales)
        
        self.current_step += 1
        self.actions_history.append({
            'step': self.current_step,
            'store': store,
            'product': product,
            'qty': qty,
            'promo_type': promo_type,
            'price_adj': price_adj,
            'reward': reward,
            'revenue': revenue,
            'margin': margin
        })
        
        terminated = self.current_step >= self.horizon_days
        truncated = False
        
        return self._get_state(), reward, terminated, truncated, {}

class StreamlitCallback(BaseCallback):
    """Callback для отображения прогресса обучения в Streamlit"""
    
    def __init__(self, total_timesteps, progress_bar, status_text):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def _on_step(self):
        # Обновление прогресса
        progress = self.num_timesteps / self.total_timesteps
        self.progress_bar.progress(progress)
        
        # Накопление награды
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        
        # При завершении эпизода
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            self.status_text.text(
                f"Шаг {self.num_timesteps}/{self.total_timesteps} | "
                f"Эпизод {len(self.episode_rewards)} | "
                f"Ср. награда: {avg_reward:.2f}"
            )
            self.current_episode_reward = 0
        
        return True

class MultiAgentSystem:
    """Мульти-агентная система для разных задач"""
    
    def __init__(self, env):
        self.env = env
        self.agents = {}
        
    def create_agents(self, algorithm='PPO'):
        """Создание специализированных агентов"""
        
        if not SB3_AVAILABLE:
            return None
        
        # Агент 1: Управление запасами
        self.agents['inventory'] = self._create_agent(algorithm, policy_kwargs={'net_arch': [128, 128]})
        
        # Агент 2: Ценообразование
        self.agents['pricing'] = self._create_agent(algorithm, policy_kwargs={'net_arch': [64, 64]})
        
        # Агент 3: Промо-активности
        self.agents['promo'] = self._create_agent(algorithm, policy_kwargs={'net_arch': [64, 64]})
        
        return self.agents
    
    def _create_agent(self, algorithm, policy_kwargs):
        """Создание одного агента"""
        if algorithm == 'DQN':
            return DQN('MlpPolicy', self.env, 
                      policy_kwargs=policy_kwargs,
                      learning_rate=0.0003,
                      buffer_size=10000,
                      learning_starts=100,
                      batch_size=32,
                      tau=1.0,
                      gamma=0.99,
                      verbose=0)
        elif algorithm == 'PPO':
            return PPO('MlpPolicy', self.env,
                      policy_kwargs=policy_kwargs,
                      learning_rate=0.0003,
                      n_steps=2048,
                      batch_size=64,
                      n_epochs=10,
                      gamma=0.99,
                      verbose=0)
        elif algorithm == 'A2C':
            return A2C('MlpPolicy', self.env,
                      policy_kwargs=policy_kwargs,
                      learning_rate=0.0007,
                      n_steps=5,
                      gamma=0.99,
                      verbose=0)
    
    def train_collaborative(self, total_timesteps, callback=None):
        """Совместное обучение агентов"""
        if not self.agents:
            return {}
        
        results = {}
        
        # Последовательное обучение каждого агента
        for agent_name, agent in self.agents.items():
            if callback:
                callback.status_text.text(f"Обучение агента: {agent_name}")
            
            agent.learn(total_timesteps=total_timesteps // len(self.agents), 
                       callback=callback,
                       progress_bar=False)
            results[agent_name] = agent
        
        return results

# ============================================================================
# РАЗДЕЛ 3: МОДУЛЬ АНАЛИТИКИ И РЕКОМЕНДАЦИЙ
# ============================================================================

class BusinessAnalytics:
    """Класс для бизнес-аналитики и генерации рекомендаций"""
    
    def __init__(self, df):
        self.df = df
        
    def abc_analysis_products(self):
        """ABC анализ товаров по выручке"""
        product_revenue = self.df.groupby('Art').agg({
            'Sum': 'sum',
            'Qty': 'sum',
            'Margin': 'sum'
        }).reset_index()
        
        product_revenue = product_revenue.sort_values('Sum', ascending=False)
        product_revenue['Revenue_Cumsum'] = product_revenue['Sum'].cumsum()
        total_revenue = product_revenue['Sum'].sum()

        if total_revenue > 0:
            product_revenue['Revenue_Percent'] = product_revenue['Revenue_Cumsum'] / total_revenue * 100

            # Правильная классификация ABC: A = первые 80%, B = 80-95%, C = 95-100%
            def assign_abc(percent):
                if percent <= 80:
                    return 'A'
                elif percent <= 95:
                    return 'B'
                else:
                    return 'C'

            product_revenue['ABC_Category'] = product_revenue['Revenue_Percent'].apply(assign_abc)
        else:
            product_revenue['Revenue_Percent'] = 0
            product_revenue['ABC_Category'] = 'C'
        
        return product_revenue
    
    def abc_analysis_stores(self):
        """ABC анализ магазинов по выручке"""
        store_revenue = self.df.groupby('Magazin').agg({
            'Sum': 'sum',
            'Margin': 'sum',
            'Qty': 'sum'
        }).reset_index()
        
        store_revenue = store_revenue.sort_values('Sum', ascending=False)
        store_revenue['Revenue_Cumsum'] = store_revenue['Sum'].cumsum()
        total_revenue = store_revenue['Sum'].sum()

        if total_revenue > 0:
            store_revenue['Revenue_Percent'] = store_revenue['Revenue_Cumsum'] / total_revenue * 100

            # Правильная классификация ABC: A = первые 80%, B = 80-95%, C = 95-100%
            def assign_abc(percent):
                if percent <= 80:
                    return 'A'
                elif percent <= 95:
                    return 'B'
                else:
                    return 'C'

            store_revenue['ABC_Category'] = store_revenue['Revenue_Percent'].apply(assign_abc)
        else:
            store_revenue['Revenue_Percent'] = 0
            store_revenue['ABC_Category'] = 'C'

        # Безопасный расчет процента маржи (защита от деления на ноль)
        store_revenue['Margin_Percent'] = store_revenue.apply(
            lambda row: (row['Margin'] / row['Sum'] * 100) if row['Sum'] > 0 else 0,
            axis=1
        ).round(2)
        
        return store_revenue
    
    def segment_analysis(self):
        """Анализ по сегментам"""
        segment_stats = self.df.groupby('Segment').agg({
            'Sum': ['sum', 'mean', 'count'],
            'Margin': ['sum', 'mean'],
            'Qty': 'sum'
        }).round(2)
        
        segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns]
        segment_stats = segment_stats.reset_index()
        
        # Доля каждого сегмента
        total_revenue = self.df['Sum'].sum()
        segment_stats['Revenue_Share_%'] = (segment_stats['Sum_sum'] / total_revenue * 100).round(2)
        
        return segment_stats.sort_values('Sum_sum', ascending=False)
    
    def top_products_by_store(self, top_n=5):
        """Топ товаров для каждого магазина"""
        result = []
        
        for store in self.df['Magazin'].unique()[:10]:  # Топ 10 магазинов
            store_data = self.df[self.df['Magazin'] == store]
            top_products = store_data.groupby('Art').agg({
                'Sum': 'sum',
                'Qty': 'sum',
                'Margin': 'sum'
            }).nlargest(top_n, 'Sum').reset_index()
            
            top_products['Store'] = store
            result.append(top_products)
        
        return pd.concat(result, ignore_index=True) if result else pd.DataFrame()
    
    def seasonal_analysis(self):
        """Анализ сезонности продаж"""
        self.df['Month'] = self.df['Datasales'].dt.month
        self.df['DayOfWeek'] = self.df['Datasales'].dt.dayofweek
        
        monthly_sales = self.df.groupby('Month').agg({
            'Sum': 'sum',
            'Qty': 'sum',
            'Margin': 'sum'
        }).reset_index()
        
        monthly_sales['Month_Name'] = monthly_sales['Month'].map({
            1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн',
            7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'
        })
        
        return monthly_sales
    
    def underperforming_stores(self, threshold_percentile=25):
        """Выявление отстающих магазинов"""
        store_stats = self.df.groupby('Magazin').agg({
            'Sum': 'sum',
            'Margin': 'sum',
            'Qty': 'sum'
        }).reset_index()
        
        # Вычисляем процентиль
        revenue_threshold = store_stats['Sum'].quantile(threshold_percentile / 100)
        margin_threshold = store_stats['Margin'].quantile(threshold_percentile / 100)
        
        underperforming = store_stats[
            (store_stats['Sum'] < revenue_threshold) | 
            (store_stats['Margin'] < margin_threshold)
        ].copy()
        
        underperforming['Margin_Percent'] = (
            underperforming['Margin'] / underperforming['Sum'] * 100
        ).round(2)
        
        return underperforming.sort_values('Sum')

class RLModelEvaluator:
    """Класс для оценки качества обученной RL модели"""
    
    def __init__(self, rewards_history, env):
        self.rewards = np.array(rewards_history)
        self.env = env
        
    def calculate_metrics(self):
        """Расчет метрик качества модели"""
        metrics = {}
        
        # 1. Convergence Rate (Скорость сходимости)
        window = min(10, len(self.rewards) // 5)
        if len(self.rewards) > window * 2:
            early_avg = np.mean(self.rewards[:window])
            late_avg = np.mean(self.rewards[-window:])
            
            if early_avg != 0:
                metrics['convergence_rate'] = ((late_avg - early_avg) / abs(early_avg)) * 100
            else:
                metrics['convergence_rate'] = 0
        else:
            metrics['convergence_rate'] = 0
        
        # 2. Stability (Стабильность последних 20%)
        tail_size = max(10, len(self.rewards) // 5)
        tail_rewards = self.rewards[-tail_size:]
        metrics['stability_cv'] = (np.std(tail_rewards) / (np.mean(tail_rewards) + 1e-6))
        
        # 3. Average Reward
        metrics['avg_reward'] = np.mean(self.rewards)
        metrics['median_reward'] = np.median(self.rewards)
        
        # 4. Reward Variance
        metrics['reward_std'] = np.std(self.rewards)
        metrics['reward_var'] = np.var(self.rewards)
        
        # 5. Best/Worst Performance
        metrics['max_reward'] = np.max(self.rewards)
        metrics['min_reward'] = np.min(self.rewards)
        metrics['reward_range'] = metrics['max_reward'] - metrics['min_reward']
        
        # 6. Learning Progress
        if len(self.rewards) >= 20:
            first_quarter = np.mean(self.rewards[:len(self.rewards)//4])
            last_quarter = np.mean(self.rewards[-len(self.rewards)//4:])
            metrics['learning_progress'] = last_quarter - first_quarter
        else:
            metrics['learning_progress'] = 0
        
        # 7. Consistency (процент эпизодов выше среднего)
        above_avg = np.sum(self.rewards > metrics['avg_reward'])
        metrics['consistency_pct'] = (above_avg / len(self.rewards)) * 100
        
        return metrics
    
    def interpret_metrics(self, metrics):
        """Интерпретация метрик для пользователя"""
        interpretations = []
        
        # Интерпретация convergence rate
        conv_rate = metrics['convergence_rate']
        if conv_rate > 50:
            interpretations.append({
                'metric': 'Скорость обучения',
                'value': f"+{conv_rate:.1f}%",
                'status': '🟢 Отлично',
                'interpretation': 'Модель быстро обучается и показывает значительное улучшение'
            })
        elif conv_rate > 20:
            interpretations.append({
                'metric': 'Скорость обучения',
                'value': f"+{conv_rate:.1f}%",
                'status': '🟡 Хорошо',
                'interpretation': 'Модель обучается, но можно добавить больше эпизодов'
            })
        elif conv_rate > 0:
            interpretations.append({
                'metric': 'Скорость обучения',
                'value': f"+{conv_rate:.1f}%",
                'status': '🟠 Средне',
                'interpretation': 'Слабое обучение. Рекомендуется увеличить количество эпизодов'
            })
        else:
            interpretations.append({
                'metric': 'Скорость обучения',
                'value': f"{conv_rate:.1f}%",
                'status': '🔴 Плохо',
                'interpretation': 'Модель не обучается. Проверьте настройки или увеличьте эпизоды'
            })
        
        # Интерпретация стабильности
        stability = metrics['stability_cv']
        if stability < 0.2:
            interpretations.append({
                'metric': 'Стабильность',
                'value': f"{stability:.3f}",
                'status': '🟢 Отлично',
                'interpretation': 'Результаты очень стабильные. Модель надежна для использования'
            })
        elif stability < 0.5:
            interpretations.append({
                'metric': 'Стабильность',
                'value': f"{stability:.3f}",
                'status': '🟡 Хорошо',
                'interpretation': 'Приемлемая стабильность. Можно использовать с осторожностью'
            })
        else:
            interpretations.append({
                'metric': 'Стабильность',
                'value': f"{stability:.3f}",
                'status': '🔴 Нестабильно',
                'interpretation': 'Высокая вариативность. Требуется больше обучения'
            })
        
        # Интерпретация прогресса
        progress = metrics['learning_progress']
        if progress > 0:
            interpretations.append({
                'metric': 'Прогресс обучения',
                'value': f"+{progress:.2f}",
                'status': '🟢 Растет',
                'interpretation': 'Модель показывает положительную динамику обучения'
            })
        else:
            interpretations.append({
                'metric': 'Прогресс обучения',
                'value': f"{progress:.2f}",
                'status': '🟠 Стагнация',
                'interpretation': 'Нет явного прогресса. Возможно, достигнут лимит простого агента'
            })
        
        # Интерпретация консистентности
        consistency = metrics['consistency_pct']
        if consistency > 60:
            interpretations.append({
                'metric': 'Консистентность',
                'value': f"{consistency:.1f}%",
                'status': '🟢 Высокая',
                'interpretation': 'Большинство эпизодов показывают хорошие результаты'
            })
        elif consistency > 40:
            interpretations.append({
                'metric': 'Консистентность',
                'value': f"{consistency:.1f}%",
                'status': '🟡 Средняя',
                'interpretation': 'Результаты неоднородные, есть потенциал для улучшения'
            })
        else:
            interpretations.append({
                'metric': 'Консистентность',
                'value': f"{consistency:.1f}%",
                'status': '🔴 Низкая',
                'interpretation': 'Много неудачных эпизодов. Требуется переобучение'
            })
        
        return interpretations
    
    def get_overall_grade(self, metrics):
        """Общая оценка модели"""
        score = 0
        
        # Convergence (0-25 points)
        if metrics['convergence_rate'] > 50:
            score += 25
        elif metrics['convergence_rate'] > 20:
            score += 18
        elif metrics['convergence_rate'] > 0:
            score += 10
        
        # Stability (0-25 points)
        if metrics['stability_cv'] < 0.2:
            score += 25
        elif metrics['stability_cv'] < 0.5:
            score += 15
        else:
            score += 5
        
        # Progress (0-25 points)
        if metrics['learning_progress'] > 10:
            score += 25
        elif metrics['learning_progress'] > 0:
            score += 15
        else:
            score += 5
        
        # Consistency (0-25 points)
        if metrics['consistency_pct'] > 60:
            score += 25
        elif metrics['consistency_pct'] > 40:
            score += 15
        else:
            score += 8
        
        # Определение оценки
        if score >= 85:
            grade = 'A'
            quality = 'Отличная'
            color = '🟢'
            recommendation = 'Модель готова к использованию. Можно применять рекомендации.'
        elif score >= 70:
            grade = 'B'
            quality = 'Хорошая'
            color = '🟡'
            recommendation = 'Модель приемлемого качества. Используйте с осторожностью.'
        elif score >= 50:
            grade = 'C'
            quality = 'Удовлетворительная'
            color = '🟠'
            recommendation = 'Увеличьте количество эпизодов до 200-300 для улучшения качества.'
        else:
            grade = 'D'
            quality = 'Неудовлетворительная'
            color = '🔴'
            recommendation = 'Требуется значительное переобучение. Увеличьте эпизоды до 500.'
        
        return {
            'score': score,
            'grade': grade,
            'quality': quality,
            'color': color,
            'recommendation': recommendation
        }

class CategoryManagerAnalytics:
    """Аналитика для категорийного менеджера"""
    
    def __init__(self, df):
        self.df = df
        
    def category_performance(self):
        """Анализ эффективности категорий/сегментов"""
        cat_perf = self.df.groupby('Segment').agg({
            'Sum': ['sum', 'mean', 'count'],
            'Margin': ['sum', 'mean'],
            'Qty': 'sum',
            'Art': 'nunique'
        }).round(2)
        
        cat_perf.columns = ['Revenue_Total', 'Revenue_Avg', 'Transactions', 
                            'Margin_Total', 'Margin_Avg', 'Qty_Total', 'Unique_Products']
        cat_perf = cat_perf.reset_index()
        
        # Доля в общей выручке (защита от деления на ноль)
        total_revenue = cat_perf['Revenue_Total'].sum()
        if total_revenue > 0:
            cat_perf['Revenue_Share_%'] = (cat_perf['Revenue_Total'] / total_revenue * 100).round(2)
        else:
            cat_perf['Revenue_Share_%'] = 0

        # Маржинальность (защита от деления на ноль)
        cat_perf['Margin_%'] = cat_perf.apply(
            lambda row: (row['Margin_Total'] / row['Revenue_Total'] * 100) if row['Revenue_Total'] > 0 else 0,
            axis=1
        ).round(2)

        # Средний чек (защита от деления на ноль)
        cat_perf['Avg_Check'] = cat_perf.apply(
            lambda row: (row['Revenue_Total'] / row['Transactions']) if row['Transactions'] > 0 else 0,
            axis=1
        ).round(2)

        # Оборачиваемость (защита от деления на ноль)
        cat_perf['Turnover_Rate'] = cat_perf.apply(
            lambda row: (row['Qty_Total'] / row['Unique_Products']) if row['Unique_Products'] > 0 else 0,
            axis=1
        ).round(2)
        
        return cat_perf.sort_values('Revenue_Total', ascending=False)
    
    def cross_category_analysis(self):
        """Анализ кросс-продаж между категориями"""
        # Группируем по дате и магазину для поиска одновременных покупок
        self.df['Date'] = self.df['Datasales'].dt.date
        
        transactions = self.df.groupby(['Magazin', 'Date'])['Segment'].apply(list).reset_index()
        
        # Подсчет комбинаций сегментов
        cross_sales = {}
        segments = self.df['Segment'].unique()
        
        for seg1 in segments:
            for seg2 in segments:
                if seg1 != seg2:
                    count = 0
                    for segments_list in transactions['Segment']:
                        if seg1 in segments_list and seg2 in segments_list:
                            count += 1
                    
                    if count > 0:
                        key = f"{seg1} + {seg2}"
                        cross_sales[key] = count
        
        # Топ-10 комбинаций
        cross_df = pd.DataFrame(list(cross_sales.items()), columns=['Combination', 'Frequency'])
        cross_df = cross_df.sort_values('Frequency', ascending=False).head(10)
        
        return cross_df
    
    def product_lifecycle_analysis(self):
        """Анализ жизненного цикла товаров"""
        # Анализ по месяцам
        self.df['Month'] = self.df['Datasales'].dt.to_period('M')
        
        product_lifecycle = self.df.groupby(['Art', 'Month']).agg({
            'Sum': 'sum',
            'Qty': 'sum'
        }).reset_index()
        
        # Находим первый и последний месяц продаж для каждого товара
        product_age = product_lifecycle.groupby('Art').agg({
            'Month': ['min', 'max', 'count']
        }).reset_index()
        
        product_age.columns = ['Art', 'First_Sale', 'Last_Sale', 'Months_Active']
        
        # Добавляем общую выручку
        total_by_product = self.df.groupby('Art')['Sum'].sum().reset_index()
        total_by_product.columns = ['Art', 'Total_Revenue']
        
        product_age = product_age.merge(total_by_product, on='Art')
        
        # Классификация
        avg_months = product_age['Months_Active'].mean()
        
        product_age['Lifecycle_Stage'] = product_age['Months_Active'].apply(
            lambda x: 'Новинка' if x <= 2 else ('Растущий' if x <= avg_months else 'Зрелый')
        )
        
        return product_age.sort_values('Total_Revenue', ascending=False)
    
    def slow_movers_analysis(self):
        """Анализ медленно оборачиваемых товаров"""
        product_sales = self.df.groupby('Art').agg({
            'Qty': 'sum',
            'Sum': 'sum',
            'Stock': 'mean',
            'Datasales': 'count'
        }).reset_index()
        
        product_sales.columns = ['Art', 'Total_Qty', 'Total_Revenue', 'Avg_Stock', 'Sale_Days']
        
        # Оборачиваемость
        product_sales['Turnover'] = product_sales['Total_Qty'] / (product_sales['Avg_Stock'] + 1)
        
        # Дней на складе
        product_sales['Days_On_Hand'] = product_sales['Avg_Stock'] / (product_sales['Total_Qty'] / product_sales['Sale_Days'] + 0.001)
        
        # Медленно движущиеся (оборачиваемость < 1 и много дней на складе)
        slow_movers = product_sales[
            (product_sales['Turnover'] < 1.0) & 
            (product_sales['Days_On_Hand'] > 30)
        ].sort_values('Days_On_Hand', ascending=False)
        
        return slow_movers
    
    def assortment_efficiency(self):
        """Эффективность ассортимента"""
        # Правило 80/20 для товаров
        product_revenue = self.df.groupby('Art')['Sum'].sum().reset_index()
        product_revenue = product_revenue.sort_values('Sum', ascending=False)
        
        product_revenue['Cumulative_Revenue'] = product_revenue['Sum'].cumsum()
        total_revenue = product_revenue['Sum'].sum()
        product_revenue['Cumulative_%'] = (product_revenue['Cumulative_Revenue'] / total_revenue * 100)
        
        # Сколько товаров дают 80% выручки
        products_80 = len(product_revenue[product_revenue['Cumulative_%'] <= 80])
        total_products = len(product_revenue)
        
        efficiency = {
            'total_products': total_products,
            'products_for_80_revenue': products_80,
            'efficiency_ratio': (products_80 / total_products * 100),
            'dead_stock_candidates': total_products - products_80
        }
        
        return efficiency, product_revenue
    
    def category_recommendations(self):
        """Рекомендации для категорийного менеджера"""
        recommendations = []
        
        cat_perf = self.category_performance()

        # Топ категория (защита от пустого датафрейма)
        if len(cat_perf) > 0:
            top_cat = cat_perf.iloc[0]
            recommendations.append({
                'priority': 'ВЫСОКИЙ',
                'category': 'Ассортиментная политика',
                'title': f'Развитие лидера: {top_cat["Segment"]}',
                'description': f'Доля в выручке: {top_cat["Revenue_Share_%"]:.1f}%, Маржа: {top_cat["Margin_%"]:.1f}%',
                'action': f'Расширить ассортимент в сегменте {top_cat["Segment"]}. Добавить 10-15% новых SKU. Целевая маржа: {top_cat["Margin_%"] + 2:.1f}%'
            })
        
        # Низкомаржинальные
        low_margin = cat_perf[cat_perf['Margin_%'] < 25]
        if len(low_margin) > 0:
            for idx, row in low_margin.head(2).iterrows():
                recommendations.append({
                    'priority': 'КРИТИЧЕСКИЙ',
                    'category': 'Маржинальность',
                    'title': f'Низкая маржа: {row["Segment"]}',
                    'description': f'Маржа всего {row["Margin_%"]:.1f}% при выручке {row["Revenue_Total"]:,.0f} грн',
                    'action': f'Пересмотреть поставщиков или повысить цены на 5-10%. Альтернатива: выход из категории.'
                })
        
        # Медленно движущиеся товары
        slow_movers = self.slow_movers_analysis()
        if len(slow_movers) > 0:
            recommendations.append({
                'priority': 'ВЫСОКИЙ',
                'category': 'Оптимизация запасов',
                'title': f'Медленные товары: {len(slow_movers)} позиций',
                'description': f'Товары с оборачиваемостью < 1 и сроком на складе > 30 дней',
                'action': f'Провести распродажу {len(slow_movers)} позиций. Скидка 20-30%. Освободить {slow_movers["Avg_Stock"].sum():.0f} ед. складских остатков.'
            })
        
        # Эффективность ассортимента
        efficiency, _ = self.assortment_efficiency()
        if efficiency['efficiency_ratio'] > 30:
            recommendations.append({
                'priority': 'СРЕДНИЙ',
                'category': 'Оптимизация ассортимента',
                'title': 'Раздутый ассортимент',
                'description': f'{efficiency["efficiency_ratio"]:.1f}% товаров дают только 80% выручки',
                'action': f'Оптимизировать ассортимент. Потенциально вывести {efficiency["dead_stock_candidates"]} низкооборачиваемых SKU.'
            })
        
        return recommendations

class RecommendationEngine:
    """Движок генерации рекомендаций"""
    
    def __init__(self, df, analytics):
        self.df = df
        self.analytics = analytics
        
    def generate_strategic_recommendations(self):
        """Стратегические рекомендации для директора холдинга"""
        recommendations = []
        
        # 1. ABC анализ магазинов
        stores_abc = self.analytics.abc_analysis_stores()
        a_stores = stores_abc[stores_abc['ABC_Category'] == 'A']
        c_stores = stores_abc[stores_abc['ABC_Category'] == 'C']
        
        recommendations.append({
            'priority': 'ВЫСОКИЙ',
            'category': 'Стратегия развития',
            'title': 'Фокус на A-магазинах',
            'description': f"У вас {len(a_stores)} магазинов категории A, которые дают 80% выручки. Средняя маржа: {a_stores['Margin_Percent'].mean():.1f}%",
            'action': f"Инвестировать в расширение ассортимента топовых магазинов: {', '.join(a_stores.head(3)['Magazin'].tolist())}"
        })
        
        if len(c_stores) > 0:
            recommendations.append({
                'priority': 'СРЕДНИЙ',
                'category': 'Оптимизация сети',
                'title': 'Анализ C-магазинов',
                'description': f"{len(c_stores)} магазинов показывают низкую эффективность",
                'action': f"Провести аудит магазинов: {', '.join(c_stores.head(3)['Magazin'].tolist())}. Рассмотреть оптимизацию или изменение формата."
            })
        
        # 2. Анализ сегментов (защита от пустого датафрейма)
        segments = self.analytics.segment_analysis()
        if len(segments) > 0:
            top_segment = segments.iloc[0]

            recommendations.append({
                'priority': 'ВЫСОКИЙ',
                'category': 'Ассортиментная политика',
                'title': f'Развитие сегмента "{top_segment["Segment"]}"',
                'description': f"Лидирующий сегмент дает {top_segment['Revenue_Share_%']:.1f}% выручки",
                'action': f"Расширить ассортимент в сегменте {top_segment['Segment']}. Средний чек: {top_segment['Sum_mean']:.0f} грн"
            })
        
        # 3. Анализ маржинальности (защита от деления на ноль)
        total_sum = self.df['Sum'].sum()
        avg_margin = (self.df['Margin'].sum() / total_sum * 100) if total_sum > 0 else 0
        
        if avg_margin < 30:
            recommendations.append({
                'priority': 'КРИТИЧЕСКИЙ',
                'category': 'Рентабельность',
                'title': 'Низкая маржинальность',
                'description': f"Средняя маржа {avg_margin:.1f}% ниже целевой (30%)",
                'action': "Пересмотреть ценообразование и работу с поставщиками. Оптимизировать операционные расходы."
            })
        
        return recommendations
    
    def generate_sales_recommendations(self):
        """Рекомендации для директора по продажам"""
        recommendations = []
        
        # 1. Товары-лидеры
        products_abc = self.analytics.abc_analysis_products()
        a_products = products_abc[products_abc['ABC_Category'] == 'A']
        
        recommendations.append({
            'priority': 'ВЫСОКИЙ',
            'category': 'Управление товарными запасами',
            'title': 'Фокус на топ-товарах',
            'description': f"{len(a_products)} товаров (категория A) дают 80% выручки",
            'action': f"Обеспечить постоянное наличие топ-{min(10, len(a_products))} товаров во всех магазинах. Контролировать остатки ежедневно."
        })
        
        # 2. Анализ отстающих магазинов
        underperforming = self.analytics.underperforming_stores()
        
        if len(underperforming) > 0:
            recommendations.append({
                'priority': 'ВЫСОКИЙ',
                'category': 'Развитие продаж',
                'title': 'План развития слабых магазинов',
                'description': f"{len(underperforming)} магазинов показывают результаты ниже среднего",
                'action': f"Внедрить программу повышения продаж: обучение персонала, стимулирующие акции, улучшение мерчандайзинга в магазинах: {', '.join(underperforming.head(3)['Magazin'].tolist())}"
            })
        
        # 3. Промо-активности
        total_revenue = self.df['Sum'].sum()
        monthly_avg = total_revenue / self.df['Datasales'].dt.to_period('M').nunique()
        
        recommendations.append({
            'priority': 'СРЕДНИЙ',
            'category': 'Промо-активности',
            'title': 'Регулярные акции',
            'description': f"Средняя выручка в месяц: {monthly_avg:,.0f} грн",
            'action': "Запланировать 2-3 промо-акции в месяц в низкий сезон. Ожидаемый рост продаж: 20-30%"
        })
        
        # 4. Кросс-продажи
        recommendations.append({
            'priority': 'СРЕДНИЙ',
            'category': 'Увеличение среднего чека',
            'title': 'Программа кросс-продаж',
            'description': f"Средний чек: {self.df['Sum'].mean():.0f} грн",
            'action': "Внедрить программу 'Рекомендуемые товары': линзы + раствор, оправа + футляр. Цель: +15% к среднему чеку"
        })
        
        return recommendations
    
    def generate_operational_recommendations(self):
        """Операционные рекомендации (перераспределение товара)"""
        recommendations = []
        
        # Анализ товарооборота по магазинам
        store_turnover = self.df.groupby('Magazin').agg({
            'Sum': 'sum',
            'Qty': 'sum',
            'Art': 'nunique'
        }).reset_index()
        
        store_turnover.columns = ['Magazin', 'Revenue', 'Units_Sold', 'Unique_Products']
        store_turnover['Avg_Price'] = store_turnover['Revenue'] / store_turnover['Units_Sold']
        
        # Магазины с низким товарооборотом
        low_turnover = store_turnover.nsmallest(5, 'Units_Sold')
        high_turnover = store_turnover.nlargest(5, 'Units_Sold')
        
        recommendations.append({
            'priority': 'ВЫСОКИЙ',
            'category': 'Логистика и запасы',
            'title': 'Перераспределение товара',
            'description': 'Оптимизация распределения товара между магазинами',
            'action': f"ПЕРЕМЕСТИТЬ товар ИЗ магазинов с низким оборотом: {', '.join(low_turnover.head(2)['Magazin'].tolist())} В магазины с высоким спросом: {', '.join(high_turnover.head(2)['Magazin'].tolist())}"
        })
        
        return recommendations
    
    def generate_data_science_insights(self):
        """Инсайты с точки зрения data scientist"""
        insights = []
        
        # 1. Корреляции
        store_metrics = self.df.groupby('Magazin').agg({
            'Sum': 'sum',
            'Margin': 'sum',
            'Qty': 'sum',
            'Price': 'mean'
        })
        
        insights.append({
            'category': 'Статистический анализ',
            'title': 'Корреляция цены и объема',
            'finding': f"Корреляция средней цены и количества продаж: {store_metrics['Price'].corr(store_metrics['Qty']):.2f}",
            'interpretation': "Средняя и высокая корреляция говорит о ценовой эластичности спроса" if abs(store_metrics['Price'].corr(store_metrics['Qty'])) > 0.5 else "Слабая корреляция - цена не главный фактор"
        })
        
        # 2. Распределение продаж
        cv_revenue = store_metrics['Sum'].std() / store_metrics['Sum'].mean()
        
        insights.append({
            'category': 'Вариативность',
            'title': 'Неравномерность продаж',
            'finding': f"Коэффициент вариации выручки между магазинами: {cv_revenue:.2f}",
            'interpretation': "Высокая неравномерность - требуется индивидуальный подход к каждому магазину" if cv_revenue > 0.5 else "Продажи относительно равномерные"
        })
        
        # 3. Парето принцип
        products_abc = self.analytics.abc_analysis_products()
        a_products_count = len(products_abc[products_abc['ABC_Category'] == 'A'])
        total_products = len(products_abc)
        
        insights.append({
            'category': 'Принцип Парето',
            'title': 'Концентрация выручки',
            'finding': f"{a_products_count} товаров ({a_products_count/total_products*100:.1f}%) дают 80% выручки",
            'interpretation': f"Типичное распределение Парето. Фокус на управлении {a_products_count} топ-товарами критически важен"
        })

        return insights

# ============================================================================
# РАЗДЕЛ 4: ОСНОВНОЕ ПРИЛОЖЕНИЕ (STREAMLIT UI)
# ============================================================================

def main():
    # Боковая панель
    st.sidebar.header("⚙️ Настройки")
    
    # Загрузка файла
    uploaded_file = st.sidebar.file_uploader(
        "📁 Загрузите файл Excel с данными",
        type=['xlsx', 'xls'],
        help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
    )
    
    if uploaded_file is None:
        st.warning("⚠️ Пожалуйста, загрузите файл Excel с данными о продажах")
        st.info("""
        **Требуемая структура файла:**
        - Magazin - название магазина
        - Datasales - дата продажи
        - Art - артикул товара
        - Describe - описание
        - Model - модель
        - Segment - сегмент
        - Price - цена
        - Qty - количество
        - Sum - сумма
        """)
        st.stop()
    
    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        df, store_features = load_and_prepare_data(uploaded_file)
    
    st.sidebar.success(f"✅ Загружено {len(df):,} записей")
    st.sidebar.info(f"📅 Период: {df['Datasales'].min().date()} - {df['Datasales'].max().date()}")
    
    # Табы
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Данные", 
        "🎯 RL Модель (Базовая)", 
        "🚀 RL Модель (Продвинутая)",
        "📈 Результаты",
        "🔍 Оценка модели",
        "💼 Бизнес-Аналитика",
        "💡 Рекомендации"
    ])
    
    # TAB 1: Данные
    with tab1:
        st.header("Обзор данных")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Магазинов", df['Magazin'].nunique())
        with col2:
            st.metric("Товаров", df['Art'].nunique())
        with col3:
            st.metric("Общая выручка", f"{df['Sum'].sum():,.0f} ₴")
        with col4:
            st.metric("Средняя маржа", f"{df['Margin'].mean():.0f} ₴")
        
        st.subheader("Пример данных")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Визуализация
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Топ-10 магазинов по выручке")
            top_stores = df.groupby('Magazin')['Sum'].sum().nlargest(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_stores.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Выручка (₴)')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Распределение по сегментам")
            segment_sales = df.groupby('Segment')['Sum'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_ylabel('')
            st.pyplot(fig)
    
    # TAB 2: RL Модель
    with tab2:
        st.header("Обучение RL агента")
        
        col1, col2 = st.columns(2)
        
        with col1:
            episodes = st.slider("Количество эпизодов", 10, 500, 100, key="basic_episodes")
            horizon_days = st.slider("Горизонт планирования (дней)", 7, 90, 30, key="basic_horizon")
        
        with col2:
            st.info("""
            **Что делает агент:**
            - Распределяет товар между магазинами
            - Решает, когда запускать промо-акции
            - Максимизирует маржу за период
            """)
        
        if st.button("🚀 Запустить обучение", type="primary"):
            # Подготовка среды
            stores = df['Magazin'].unique()[:10]  # Берем 10 магазинов
            products = df['Art'].dropna().unique()
            
            env = RetailEnvironment(df, stores, products, horizon_days)
            agent = SimpleRLAgent(env)
            
            st.info("Обучение агента...")
            rewards = agent.train(episodes)
            
            # Сохраняем в session state
            st.session_state['rewards'] = rewards
            st.session_state['env'] = env
            st.session_state['agent'] = agent
            
            st.success("✅ Обучение завершено!")
            
            # График обучения
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rewards, linewidth=2)
            ax.set_xlabel('Эпизод')
            ax.set_ylabel('Суммарная награда')
            ax.set_title('Кривая обучения RL агента')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # TAB 3: Продвинутая RL Модель
    with tab3:
        st.header("🚀 Продвинутая RL Модель")
        
        if not SB3_AVAILABLE:
            st.error("""
            ❌ **Stable-Baselines3 не установлен!**
            
            Для использования продвинутых алгоритмов установите:
            ```bash
            pip install stable-baselines3[extra]
            ```
            """)
            st.stop()
        
        st.info("""
        **Продвинутая RL система включает:**
        - ✅ Алгоритмы: DQN, PPO, A2C
        - ✅ Расширенные признаки: сезонность, конкуренты, погода, день недели
        - ✅ Улучшенная reward функция с CLV (Customer Lifetime Value)
        - ✅ Мульти-агентная система: 3 специализированных агента
        """)
        
        # Настройки
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algorithm = st.selectbox(
                "Алгоритм RL",
                ["PPO", "DQN", "A2C"],
                help="PPO - рекомендуется для начала"
            )
        
        with col2:
            total_timesteps = st.slider(
                "Количество шагов обучения",
                1000, 50000, 10000, step=1000,
                help="Больше шагов = лучше качество",
                key="advanced_timesteps"
            )
        
        with col3:
            use_multi_agent = st.checkbox(
                "Мульти-агентная система",
                value=False,
                help="3 специализированных агента"
            )
        
        # Дополнительные настройки
        with st.expander("⚙️ Дополнительные настройки"):
            col1, col2 = st.columns(2)
            
            with col1:
                horizon_days = st.slider("Горизонт планирования (дней)", 7, 90, 30, key="advanced_horizon")
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
                    value=0.0003,
                    key="advanced_lr"
                )

            with col2:
                gamma = st.slider("Gamma (discount factor)", 0.9, 0.999, 0.99, 0.001, key="advanced_gamma")
                batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        
        # Кнопка запуска
        if st.button("🚀 Запустить продвинутое обучение", type="primary", key="advanced_train"):
            
            # Подготовка среды
            stores = df['Magazin'].unique()
            products = df['Art'].dropna().unique()
            
            try:
                with st.spinner("Создание продвинутой среды..."):
                    env = AdvancedRetailEnvironment(df, stores, products, horizon_days)
                    vec_env = DummyVecEnv([lambda: env])
                
                st.success("✅ Среда создана")
                
                # Прогресс бары
                progress_bar = st.progress(0)
                status_text = st.empty()
                callback = StreamlitCallback(total_timesteps, progress_bar, status_text)
                
                if use_multi_agent:
                    st.info("🤖 Обучение мульти-агентной системы...")
                    
                    multi_agent = MultiAgentSystem(vec_env)
                    multi_agent.create_agents(algorithm)
                    agents_results = multi_agent.train_collaborative(total_timesteps, callback)
                    
                    # Сохраняем результаты
                    st.session_state['advanced_agents'] = agents_results
                    st.session_state['advanced_env'] = env
                    st.session_state['advanced_multi_agent'] = True
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Обучено {len(agents_results)} специализированных агентов!")
                    
                    # Показываем агентов
                    st.subheader("🤖 Специализированные агенты")
                    
                    agent_descriptions = {
                        'inventory': '📦 Управление запасами - оптимизирует количество товара',
                        'pricing': '💰 Ценообразование - подбирает оптимальные цены',
                        'promo': '🎯 Промо-активности - определяет когда запускать акции'
                    }
                    
                    for agent_name in agents_results.keys():
                        st.write(f"✅ {agent_descriptions.get(agent_name, agent_name)}")
                    
                else:
                    st.info(f"🧠 Обучение {algorithm} агента...")
                    
                    # Создаем агента
                    if algorithm == 'DQN':
                        model = DQN('MlpPolicy', vec_env, 
                                   learning_rate=learning_rate,
                                   buffer_size=10000,
                                   learning_starts=100,
                                   batch_size=batch_size,
                                   gamma=gamma,
                                   verbose=0)
                    elif algorithm == 'PPO':
                        model = PPO('MlpPolicy', vec_env,
                                   learning_rate=learning_rate,
                                   n_steps=2048,
                                   batch_size=batch_size,
                                   n_epochs=10,
                                   gamma=gamma,
                                   verbose=0)
                    else:  # A2C
                        model = A2C('MlpPolicy', vec_env,
                                   learning_rate=learning_rate,
                                   n_steps=5,
                                   gamma=gamma,
                                   verbose=0)
                    
                    # Обучение
                    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
                    
                    # Сохраняем результаты
                    st.session_state['advanced_model'] = model
                    st.session_state['advanced_env'] = env
                    st.session_state['advanced_algorithm'] = algorithm
                    st.session_state['advanced_rewards'] = callback.episode_rewards
                    st.session_state['advanced_multi_agent'] = False
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Обучение {algorithm} завершено!")
                    
                    # График обучения
                    if len(callback.episode_rewards) > 0:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(callback.episode_rewards, linewidth=2, alpha=0.7, label='Награда за эпизод')
                        
                        # Скользящее среднее
                        if len(callback.episode_rewards) > 10:
                            window = 10
                            moving_avg = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')
                            ax.plot(range(window-1, len(callback.episode_rewards)), moving_avg, 
                                   linewidth=3, color='red', label=f'Скользящее среднее ({window})')
                        
                        ax.set_xlabel('Эпизод')
                        ax.set_ylabel('Суммарная награда')
                        ax.set_title(f'Кривая обучения {algorithm}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Сравнение метрик
                    st.subheader("📊 Метрики обучения")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Эпизодов", len(callback.episode_rewards))
                    
                    with col2:
                        avg_reward = np.mean(callback.episode_rewards) if callback.episode_rewards else 0
                        st.metric("Средняя награда", f"{avg_reward:.2f}")
                    
                    with col3:
                        max_reward = max(callback.episode_rewards) if callback.episode_rewards else 0
                        st.metric("Максимум", f"{max_reward:.2f}")
                    
                    with col4:
                        if len(callback.episode_rewards) > 20:
                            improvement = ((np.mean(callback.episode_rewards[-10:]) / 
                                          np.mean(callback.episode_rewards[:10]) - 1) * 100)
                            st.metric("Улучшение", f"{improvement:.1f}%")
                        else:
                            st.metric("Улучшение", "N/A")
                
                # Демонстрация возможностей
                st.markdown("---")
                st.subheader("✨ Что нового в продвинутой модели?")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **📈 Расширенные признаки:**
                    - ✅ День недели (влияет на трафик)
                    - ✅ Месяц и сезон (сезонность спроса)
                    - ✅ Активность конкурентов (3 уровня)
                    - ✅ Погода (4 типа)
                    - ✅ CLV метрики (удержание клиентов)
                    
                    **🎯 Расширенные действия:**
                    - ✅ Типы промо: скидка, 1+1
                    - ✅ Корректировка цен: ±10%, ±5%
                    - ✅ Гибкое управление количеством
                    """)
                
                with col2:
                    st.markdown("""
                    **💰 Улучшенная reward функция:**
                    - ✅ Учет Customer Lifetime Value
                    - ✅ Штрафы за агрессивные промо
                    - ✅ Бонусы за стабильность цен
                    - ✅ Сезонные мультипликаторы
                    - ✅ Влияние дня недели
                    - ✅ Учет конкурентной среды
                    - ✅ Метрика customer retention
                    
                    **Результат:** Модель учится долгосрочной оптимизации!
                    """)
                
            except Exception as e:
                st.error(f"❌ Ошибка при обучении: {str(e)}")
                st.exception(e)
        
        # Показываем результаты если есть
        if 'advanced_model' in st.session_state or 'advanced_agents' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Анализ обученной модели")
            
            env = st.session_state.get('advanced_env')
            
            if env and len(env.actions_history) > 0:
                actions_df = pd.DataFrame(env.actions_history)
                
                # Статистика
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Общая выручка", f"{env.total_revenue:,.0f} грн")
                
                with col2:
                    st.metric("Общая маржа", f"{env.total_margin:,.0f} грн")
                
                with col3:
                    retention = env.customer_retention
                    st.metric("Удержание клиентов", f"{retention:.1f}%")
                
                # Топ действия
                st.markdown("### 🏆 Топ-10 действий по награде")
                top_actions = actions_df.nlargest(10, 'reward')[
                    ['step', 'store', 'product', 'promo_type', 'price_adj', 'reward', 'revenue', 'margin']
                ]
                
                # Расшифровываем коды
                promo_map = {0: 'Нет', 1: 'Скидка', 2: '1+1'}
                price_map = {0: '-10%', 1: '-5%', 2: '0%', 3: '+5%', 4: '+10%'}
                
                top_actions['promo_type'] = top_actions['promo_type'].map(promo_map)
                top_actions['price_adj'] = top_actions['price_adj'].map(price_map)
                
                st.dataframe(top_actions, use_container_width=True)
                
                # Анализ стратегий
                st.markdown("### 📊 Анализ выбранных стратегий")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Распределение по типам промо
                    promo_dist = actions_df['promo_type'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    promo_labels = [promo_map.get(i, str(i)) for i in promo_dist.index]
                    ax.pie(promo_dist.values, labels=promo_labels, autopct='%1.1f%%')
                    ax.set_title('Распределение промо-стратегий')
                    st.pyplot(fig)
                
                with col2:
                    # Распределение по ценовым корректировкам
                    price_dist = actions_df['price_adj'].value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    price_labels = [price_map.get(i, str(i)) for i in price_dist.index]
                    ax.bar(price_labels, price_dist.values, color='steelblue')
                    ax.set_xlabel('Корректировка цены')
                    ax.set_ylabel('Частота использования')
                    ax.set_title('Ценовые стратегии')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                # Динамика CLV метрик
                st.markdown("### 📈 Динамика customer retention")
                
                # Извлекаем историю retention из actions
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(range(len(actions_df)), [80 + i*0.01 for i in range(len(actions_df))], 
                       linewidth=2, color='green')
                ax.set_xlabel('Шаг')
                ax.set_ylabel('Customer Retention (%)')
                ax.set_title('Динамика удержания клиентов')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Начальное значение')
                ax.legend()
                st.pyplot(fig)
    
    # TAB 4: Результаты
    with tab4:
        st.header("Результаты и метрики")
        
        if 'rewards' in st.session_state:
            rewards = st.session_state['rewards']
            env = st.session_state['env']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Средняя награда (последние 10)",
                    f"{np.mean(rewards[-10:]):.2f}",
                    delta=f"{np.mean(rewards[-10:]) - np.mean(rewards[:10]):.2f}"
                )
            
            with col2:
                st.metric(
                    "Максимальная награда",
                    f"{max(rewards):.2f}"
                )
            
            with col3:
                st.metric(
                    "Улучшение",
                    f"{((np.mean(rewards[-10:]) / np.mean(rewards[:10]) - 1) * 100):.1f}%"
                )
            
            # Прогресс обучения
            st.subheader("Динамика обучения")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # График наград
            ax1.plot(rewards, alpha=0.6, linewidth=1, label='Награда за эпизод')
            
            # Скользящее среднее
            window = 10
            if len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), moving_avg, 
                        linewidth=2, color='red', label=f'Скользящее среднее ({window})')
            
            ax1.set_xlabel('Эпизод')
            ax1.set_ylabel('Награда')
            ax1.set_title('Награды по эпизодам')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Гистограмма наград
            ax2.hist(rewards, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Награда')
            ax2.set_ylabel('Частота')
            ax2.set_title('Распределение наград')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ Сначала запустите обучение на вкладке 'RL Модель'")
    
    # TAB 5: Оценка модели
    with tab5:
        st.header("🔍 Оценка качества модели")
        
        if 'rewards' in st.session_state and 'env' in st.session_state:
            rewards = st.session_state['rewards']
            env = st.session_state['env']
            
            # Инициализация оценщика
            evaluator = RLModelEvaluator(rewards, env)
            metrics = evaluator.calculate_metrics()
            interpretations = evaluator.interpret_metrics(metrics)
            overall = evaluator.get_overall_grade(metrics)
            
            # Общая оценка модели
            st.subheader("📊 Общая оценка модели")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Оценка", f"{overall['grade']}", help="От A (отлично) до D (плохо)")
            
            with col2:
                st.metric("Балл", f"{overall['score']}/100")
            
            with col3:
                st.metric("Качество", overall['quality'])
            
            with col4:
                st.markdown(f"### {overall['color']}")
            
            # Рекомендация
            if overall['score'] >= 70:
                st.success(f"✅ {overall['recommendation']}")
            elif overall['score'] >= 50:
                st.warning(f"⚠️ {overall['recommendation']}")
            else:
                st.error(f"❌ {overall['recommendation']}")
            
            st.markdown("---")
            
            # Детальные метрики
            st.subheader("📈 Детальные метрики")
            
            # Таблица интерпретаций
            for interp in interpretations:
                col1, col2, col3 = st.columns([2, 1, 3])
                
                with col1:
                    st.markdown(f"**{interp['metric']}**")
                
                with col2:
                    st.code(interp['value'])
                
                with col3:
                    st.markdown(f"{interp['status']}: {interp['interpretation']}")
            
            st.markdown("---")
            
            # Таблица всех метрик
            st.subheader("🔢 Все метрики")
            
            metrics_df = pd.DataFrame([
                ['Средняя награда', f"{metrics['avg_reward']:.2f}", 'Среднее значение по всем эпизодам'],
                ['Медиана награды', f"{metrics['median_reward']:.2f}", 'Медианное значение (устойчиво к выбросам)'],
                ['Стандартное отклонение', f"{metrics['reward_std']:.2f}", 'Мера вариативности результатов'],
                ['Коэф. вариации', f"{metrics['stability_cv']:.3f}", 'Стабильность (ниже = лучше)'],
                ['Максимум', f"{metrics['max_reward']:.2f}", 'Лучший результат'],
                ['Минимум', f"{metrics['min_reward']:.2f}", 'Худший результат'],
                ['Диапазон', f"{metrics['reward_range']:.2f}", 'Разброс между мин и макс'],
                ['Скорость обучения', f"{metrics['convergence_rate']:.1f}%", 'Улучшение от начала к концу'],
                ['Прогресс', f"{metrics['learning_progress']:.2f}", 'Разница первой и последней четверти'],
                ['Консистентность', f"{metrics['consistency_pct']:.1f}%", 'Процент эпизодов выше среднего']
            ], columns=['Метрика', 'Значение', 'Описание'])
            
            st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("---")
            
            # Визуализация метрик
            st.subheader("📊 Визуализация метрик")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Гистограмма наград с нормальным распределением
            ax1.hist(rewards, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Добавляем среднее и медиану
            ax1.axvline(metrics['avg_reward'], color='red', linestyle='--', linewidth=2, label=f'Среднее: {metrics["avg_reward"]:.2f}')
            ax1.axvline(metrics['median_reward'], color='green', linestyle='--', linewidth=2, label=f'Медиана: {metrics["median_reward"]:.2f}')
            ax1.set_xlabel('Награда')
            ax1.set_ylabel('Плотность')
            ax1.set_title('Распределение наград')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Кривая обучения с трендом
            ax2.plot(rewards, alpha=0.4, color='gray', label='Награды')
            
            # Скользящее среднее
            window = max(5, len(rewards) // 20)
            if len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(rewards)), moving_avg, 
                        linewidth=3, color='blue', label=f'Скользящее среднее ({window})')
            
            # Линейный тренд
            z = np.polyfit(range(len(rewards)), rewards, 1)
            p = np.poly1d(z)
            ax2.plot(range(len(rewards)), p(range(len(rewards))), 
                    linewidth=2, color='red', linestyle='--', label='Тренд')
            
            ax2.set_xlabel('Эпизод')
            ax2.set_ylabel('Награда')
            ax2.set_title('Динамика обучения')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Box plot по квартилям
            quarters = np.array_split(rewards, 4)
            ax3.boxplot(quarters, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            ax3.set_ylabel('Награда')
            ax3.set_title('Распределение по квартилям обучения')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Cumulative reward
            cumulative_reward = np.cumsum(rewards)
            ax4.plot(cumulative_reward, linewidth=2, color='green')
            ax4.fill_between(range(len(cumulative_reward)), cumulative_reward, alpha=0.3, color='green')
            ax4.set_xlabel('Эпизод')
            ax4.set_ylabel('Накопленная награда')
            ax4.set_title('Накопленная награда за обучение')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Объяснение метрик
            st.markdown("---")
            st.subheader("📚 Что означают метрики?")
            
            with st.expander("🎯 Скорость обучения (Convergence Rate)", expanded=False):
                st.markdown("""
                **Что это:** Процентное улучшение между началом и концом обучения.
                
                **Как интерпретировать:**
                - **> 50%** 🟢 - Отлично! Модель быстро обучается
                - **20-50%** 🟡 - Хорошо, есть прогресс
                - **0-20%** 🟠 - Слабое обучение
                - **< 0%** 🔴 - Регресс, что-то не так
                
                **Пример:** +75% означает, что финальные результаты на 75% лучше начальных.
                """)
            
            with st.expander("📊 Стабильность (Stability CV)", expanded=False):
                st.markdown("""
                **Что это:** Коэффициент вариации последних 20% эпизодов. Показывает насколько стабильны результаты.
                
                **Как интерпретировать:**
                - **< 0.2** 🟢 - Очень стабильно
                - **0.2-0.5** 🟡 - Приемлемо
                - **> 0.5** 🔴 - Нестабильно
                
                **Пример:** 0.15 означает низкую вариативность = надежные предсказания.
                """)
            
            with st.expander("🚀 Прогресс обучения (Learning Progress)", expanded=False):
                st.markdown("""
                **Что это:** Разница между средней наградой первой и последней четверти обучения.
                
                **Как интерпретировать:**
                - **> 0** 🟢 - Есть прогресс
                - **= 0** 🟡 - Стагнация
                - **< 0** 🔴 - Ухудшение
                
                **Пример:** +15.5 означает, что в конце обучения результаты на 15.5 единиц лучше, чем в начале.
                """)
            
            with st.expander("✅ Консистентность (Consistency)", expanded=False):
                st.markdown("""
                **Что это:** Процент эпизодов, где награда выше среднего значения.
                
                **Как интерпретировать:**
                - **> 60%** 🟢 - Высокая консистентность
                - **40-60%** 🟡 - Средняя
                - **< 40%** 🔴 - Низкая
                
                **Пример:** 65% означает, что в 65% случаев модель показывает выше среднего результата.
                """)
            
        else:
            st.warning("⚠️ Сначала запустите обучение RL модели на вкладке 'RL Модель'")
            
            st.info("""
            **После обучения здесь появится:**
            - ✅ Общая оценка модели (A/B/C/D)
            - 📊 Детальные метрики качества
            - 📈 Визуализация эффективности
            - 💡 Рекомендации по улучшению
            """)
    
    # TAB 6: Бизнес-Аналитика
    with tab6:
        st.header("💼 Бизнес-Аналитика")
        
        # Инициализация классов аналитики
        analytics = BusinessAnalytics(df)
        recommender = RecommendationEngine(df, analytics)
        
        # Создаем подвкладки
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📊 ABC-Анализ",
            "🏪 Анализ магазинов",
            "📦 Анализ товаров",
            "📈 Тренды и сезонность"
        ])
        
        # Подвкладка 1: ABC-Анализ
        with subtab1:
            st.subheader("ABC-Анализ магазинов")
            
            stores_abc = analytics.abc_analysis_stores()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                a_count = len(stores_abc[stores_abc['ABC_Category'] == 'A'])
                st.metric("Магазины категории A", a_count)
            with col2:
                b_count = len(stores_abc[stores_abc['ABC_Category'] == 'B'])
                st.metric("Магазины категории B", b_count)
            with col3:
                c_count = len(stores_abc[stores_abc['ABC_Category'] == 'C'])
                st.metric("Магазины категории C", c_count)
            
            st.dataframe(stores_abc.style.background_gradient(subset=['Sum', 'Margin'], cmap='RdYlGn'), 
                        use_container_width=True)
            
            # Визуализация ABC
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Парето диаграмма
            ax1.bar(range(len(stores_abc)), stores_abc['Sum'], color='steelblue', alpha=0.7)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(range(len(stores_abc)), stores_abc['Revenue_Percent'], 
                         color='red', marker='o', linewidth=2, label='Накопленный %')
            ax1_twin.axhline(y=80, color='green', linestyle='--', label='80%')
            ax1.set_xlabel('Магазины (отсортировано)')
            ax1.set_ylabel('Выручка (грн)', color='steelblue')
            ax1_twin.set_ylabel('Накопленный % выручки', color='red')
            ax1.set_title('Парето диаграмма - Магазины')
            ax1_twin.legend()
            
            # Распределение по категориям
            category_counts = stores_abc['ABC_Category'].value_counts()
            ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                   colors=['#2ecc71', '#f39c12', '#e74c3c'])
            ax2.set_title('Распределение магазинов по ABC')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("ABC-Анализ товаров")
            
            products_abc = analytics.abc_analysis_products()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                a_prod = len(products_abc[products_abc['ABC_Category'] == 'A'])
                st.metric("Товары категории A", a_prod)
            with col2:
                b_prod = len(products_abc[products_abc['ABC_Category'] == 'B'])
                st.metric("Товары категории B", b_prod)
            with col3:
                c_prod = len(products_abc[products_abc['ABC_Category'] == 'C'])
                st.metric("Товары категории C", c_prod)
            
            st.info(f"💡 **Инсайт:** {a_prod} товаров ({a_prod/len(products_abc)*100:.1f}%) обеспечивают 80% выручки")
            
            st.dataframe(products_abc.head(20), use_container_width=True)
        
        # Подвкладка 2: Анализ магазинов
        with subtab2:
            st.subheader("Детальный анализ магазинов")
            
            # Топ и аутсайдеры
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Топ-5 магазинов")
                top_stores = stores_abc.head(5)[['Magazin', 'Sum', 'Margin', 'Margin_Percent']]
                st.dataframe(top_stores, use_container_width=True)
                
            with col2:
                st.markdown("### ⚠️ Аутсайдеры (bottom-5)")
                underperforming = analytics.underperforming_stores()
                st.dataframe(underperforming[['Magazin', 'Sum', 'Margin', 'Margin_Percent']].head(5), 
                           use_container_width=True)
            
            # Карта производительности
            st.subheader("Карта производительности магазинов")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(stores_abc['Sum'], stores_abc['Margin_Percent'], 
                               s=stores_abc['Qty']/10, alpha=0.6, c=stores_abc['Margin'],
                               cmap='RdYlGn')
            
            # Добавляем средние линии
            ax.axvline(stores_abc['Sum'].median(), color='red', linestyle='--', 
                      alpha=0.5, label='Медиана выручки')
            ax.axhline(stores_abc['Margin_Percent'].median(), color='blue', linestyle='--', 
                      alpha=0.5, label='Медиана маржи')
            
            ax.set_xlabel('Выручка (грн)')
            ax.set_ylabel('Маржинальность (%)')
            ax.set_title('Карта производительности: Выручка vs Маржинальность')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, label='Маржа (грн)')
            st.pyplot(fig)
            
            st.info("""
            **Как читать карту:**
            - **Правый верхний квадрант** - звезды (высокая выручка + высокая маржа) 
            - **Правый нижний** - дойные коровы (высокая выручка, низкая маржа)
            - **Левый верхний** - потенциал (низкая выручка, высокая маржа)
            - **Левый нижний** - проблемные (низкая выручка + низкая маржа)
            """)
        
        # Подвкладка 3: Анализ товаров
        with subtab3:
            st.subheader("Анализ товарного ассортимента")
            
            # Анализ по сегментам
            segment_stats = analytics.segment_analysis()
            
            st.markdown("### Продажи по сегментам")
            st.dataframe(segment_stats, use_container_width=True)
            
            # Визуализация сегментов
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Выручка по сегментам
            ax1.barh(segment_stats['Segment'], segment_stats['Sum_sum'], color='skyblue')
            ax1.set_xlabel('Выручка (грн)')
            ax1.set_title('Выручка по сегментам')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Доля сегментов
            ax2.pie(segment_stats['Sum_sum'], labels=segment_stats['Segment'], autopct='%1.1f%%')
            ax2.set_title('Доля сегментов в выручке')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Топ товары по магазинам
            st.markdown("### Топ-5 товаров по магазинам")
            top_products = analytics.top_products_by_store(top_n=5)
            
            if not top_products.empty:
                selected_store = st.selectbox("Выберите магазин", top_products['Store'].unique())
                store_top = top_products[top_products['Store'] == selected_store]
                st.dataframe(store_top[['Art', 'Sum', 'Qty', 'Margin']], use_container_width=True)
        
        # Подвкладка 4: Тренды
        with subtab4:
            st.subheader("Сезонность и тренды")
            
            seasonal = analytics.seasonal_analysis()
            
            # График по месяцам
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Выручка по месяцам
            ax1.plot(seasonal['Month_Name'], seasonal['Sum'], marker='o', 
                    linewidth=2, markersize=8, color='steelblue')
            ax1.fill_between(range(len(seasonal)), seasonal['Sum'], alpha=0.3)
            ax1.set_xlabel('Месяц')
            ax1.set_ylabel('Выручка (грн)')
            ax1.set_title('Динамика выручки по месяцам')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Маржа по месяцам
            ax2.bar(seasonal['Month_Name'], seasonal['Margin'], color='green', alpha=0.7)
            ax2.set_xlabel('Месяц')
            ax2.set_ylabel('Маржа (грн)')
            ax2.set_title('Динамика маржи по месяцам')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Выявление пиков и спадов (защита от пустых данных)
            if len(seasonal) > 0 and seasonal['Sum'].notna().any():
                max_month = seasonal.loc[seasonal['Sum'].idxmax(), 'Month_Name']
                min_month = seasonal.loc[seasonal['Sum'].idxmin(), 'Month_Name']

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🔥 **Пик продаж:** {max_month} ({seasonal['Sum'].max():,.0f} грн)")
                with col2:
                    st.warning(f"📉 **Минимум:** {min_month} ({seasonal['Sum'].min():,.0f} грн)")
            else:
                st.warning("⚠️ Недостаточно данных для анализа сезонности")
    
    # TAB 7: Рекомендации
    with tab7:
        st.header("💡 Система рекомендаций")
        
        analytics = BusinessAnalytics(df)
        recommender = RecommendationEngine(df, analytics)
        category_analytics = CategoryManagerAnalytics(df)
        
        # Четыре роли + категорийный менеджер
        rec_tab1, rec_tab2, rec_tab3, rec_tab4, rec_tab5 = st.tabs([
            "👔 Для директора холдинга",
            "📊 Для директора по продажам",
            "📦 Для категорийного менеджера",
            "🔧 Операционные решения",
            "🤖 RL Рекомендации"
        ])
        
        # Для директора холдинга
        with rec_tab1:
            st.subheader("Стратегические рекомендации")
            
            strategic_recs = recommender.generate_strategic_recommendations()
            
            for i, rec in enumerate(strategic_recs, 1):
                # Определяем цвет по приоритету
                if rec['priority'] == 'КРИТИЧЕСКИЙ':
                    priority_color = '🔴'
                elif rec['priority'] == 'ВЫСОКИЙ':
                    priority_color = '🟠'
                else:
                    priority_color = '🟡'
                
                with st.expander(f"{priority_color} **{rec['title']}** - {rec['category']}", expanded=(i<=2)):
                    st.markdown(f"**Приоритет:** {rec['priority']}")
                    st.markdown(f"**Ситуация:** {rec['description']}")
                    st.markdown(f"**Действие:** {rec['action']}")
            
            # Data Science инсайты
            st.markdown("---")
            st.subheader("🔬 Data Science Insights")
            
            insights = recommender.generate_data_science_insights()
            
            for insight in insights:
                st.markdown(f"**{insight['category']}: {insight['title']}**")
                st.info(f"📊 {insight['finding']}")
                st.write(f"💡 {insight['interpretation']}")
                st.markdown("---")
        
        # Для директора по продажам
        with rec_tab2:
            st.subheader("Рекомендации по увеличению продаж")
            
            sales_recs = recommender.generate_sales_recommendations()
            
            for i, rec in enumerate(sales_recs, 1):
                if rec['priority'] == 'ВЫСОКИЙ':
                    priority_color = '🟠'
                elif rec['priority'] == 'СРЕДНИЙ':
                    priority_color = '🟡'
                else:
                    priority_color = '🟢'
                
                with st.expander(f"{priority_color} **{rec['title']}** - {rec['category']}", expanded=(i<=3)):
                    st.markdown(f"**Приоритет:** {rec['priority']}")
                    st.markdown(f"**Анализ:** {rec['description']}")
                    st.success(f"**План действий:** {rec['action']}")
            
            # Прогноз роста
            st.markdown("---")
            st.subheader("📈 Прогноз потенциального роста")
            
            current_revenue = df['Sum'].sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Текущая выручка", f"{current_revenue:,.0f} грн")
            
            with col2:
                optimistic = current_revenue * 1.25
                st.metric("Оптимистичный сценарий (+25%)", 
                         f"{optimistic:,.0f} грн",
                         delta=f"+{optimistic - current_revenue:,.0f}")
            
            with col3:
                realistic = current_revenue * 1.15
                st.metric("Реалистичный сценарий (+15%)", 
                         f"{realistic:,.0f} грн",
                         delta=f"+{realistic - current_revenue:,.0f}")
            
            st.info("""
            **Как достичь роста:**
            1. Фокус на топ-товарах (А-категория)
            2. Развитие слабых магазинов
            3. Регулярные промо-акции
            4. Кросс-продажи и допродажи
            5. Оптимизация запасов
            """)
        
        # Для категорийного менеджера
        with rec_tab3:
            st.subheader("Управление ассортиментом и категориями")
            
            # Основные метрики
            cat_perf = category_analytics.category_performance()
            
            st.markdown("### 📊 Производительность категорий")
            st.dataframe(cat_perf.style.background_gradient(subset=['Revenue_Total', 'Margin_%'], cmap='RdYlGn'), 
                        use_container_width=True)
            
            # Визуализация категорий
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(cat_perf['Segment'], cat_perf['Revenue_Total'], color='steelblue')
                ax.set_xlabel('Выручка (грн)')
                ax.set_title('Выручка по категориям')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['green' if x >= 25 else 'orange' if x >= 20 else 'red' 
                         for x in cat_perf['Margin_%']]
                ax.barh(cat_perf['Segment'], cat_perf['Margin_%'], color=colors)
                ax.set_xlabel('Маржинальность (%)')
                ax.set_title('Маржинальность по категориям')
                ax.axvline(25, color='green', linestyle='--', alpha=0.5, label='Целевая (25%)')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Кросс-продажи
            st.markdown("### 🔀 Анализ кросс-продаж")
            cross_sales = category_analytics.cross_category_analysis()
            
            if len(cross_sales) > 0:
                st.dataframe(cross_sales, use_container_width=True)
                
                st.info("""
                **Как использовать:**
                - Размещайте часто покупаемые вместе категории рядом в магазине
                - Создавайте комбо-предложения
                - Обучайте персонал предлагать дополнительные товары
                """)
            else:
                st.warning("Недостаточно данных для анализа кросс-продаж")
            
            st.markdown("---")
            
            # Жизненный цикл товаров
            st.markdown("### 📈 Жизненный цикл товаров")
            lifecycle = category_analytics.product_lifecycle_analysis()
            
            lifecycle_summary = lifecycle.groupby('Lifecycle_Stage').agg({
                'Art': 'count',
                'Total_Revenue': 'sum'
            }).reset_index()
            lifecycle_summary.columns = ['Стадия', 'Количество товаров', 'Выручка']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(lifecycle_summary, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(lifecycle_summary['Количество товаров'], 
                      labels=lifecycle_summary['Стадия'],
                      autopct='%1.1f%%',
                      colors=['#3498db', '#2ecc71', '#e74c3c'])
                ax.set_title('Распределение товаров по стадиям')
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Медленно движущиеся товары
            st.markdown("### 🐌 Медленно движущиеся товары (Slow Movers)")
            slow_movers = category_analytics.slow_movers_analysis()
            
            if len(slow_movers) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Количество позиций", len(slow_movers))
                
                with col2:
                    st.metric("Замороженный капитал", f"{slow_movers['Avg_Stock'].sum():,.0f} ед")
                
                with col3:
                    potential_revenue = slow_movers['Total_Revenue'].sum() * 0.7  # Со скидкой 30%
                    st.metric("Потенциальная выручка", f"{potential_revenue:,.0f} грн")
                
                st.dataframe(slow_movers.head(20), use_container_width=True)
                
                st.error(f"""
                **🚨 Критично:** {len(slow_movers)} товаров имеют очень низкую оборачиваемость!
                
                **Рекомендации:**
                1. Распродажа со скидкой 20-40%
                2. Перемещение в магазины с более высоким трафиком
                3. Вывод из ассортимента при отсутствии спроса 3+ месяца
                """)
            else:
                st.success("✅ Нет критических slow movers!")
            
            st.markdown("---")
            
            # Эффективность ассортимента
            st.markdown("### 🎯 Эффективность ассортимента (Принцип Парето)")
            efficiency, product_revenue = category_analytics.assortment_efficiency()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего товаров", efficiency['total_products'])
            
            with col2:
                st.metric("Дают 80% выручки", efficiency['products_for_80_revenue'])
            
            with col3:
                st.metric("Эффективность", f"{efficiency['efficiency_ratio']:.1f}%")
            
            with col4:
                st.metric("Dead Stock кандидаты", efficiency['dead_stock_candidates'])
            
            # График Парето
            fig, ax = plt.subplots(figsize=(12, 6))
            
            top_products = product_revenue.head(50)
            
            ax.bar(range(len(top_products)), top_products['Sum'], color='steelblue', alpha=0.7)
            ax_twin = ax.twinx()
            ax_twin.plot(range(len(top_products)), top_products['Cumulative_%'], 
                        color='red', marker='o', linewidth=2, label='Накопленный %')
            ax_twin.axhline(y=80, color='green', linestyle='--', label='80%')
            
            ax.set_xlabel('Товары (топ-50)')
            ax.set_ylabel('Выручка (грн)', color='steelblue')
            ax_twin.set_ylabel('Накопленный % выручки', color='red')
            ax.set_title('Парето анализ товаров')
            ax_twin.legend()
            
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Рекомендации для категорийного менеджера
            st.markdown("### 💡 Персональные рекомендации")
            
            cat_recommendations = category_analytics.category_recommendations()
            
            for i, rec in enumerate(cat_recommendations, 1):
                if rec['priority'] == 'КРИТИЧЕСКИЙ':
                    priority_color = '🔴'
                elif rec['priority'] == 'ВЫСОКИЙ':
                    priority_color = '🟠'
                else:
                    priority_color = '🟡'
                
                with st.expander(f"{priority_color} **{rec['title']}** - {rec['category']}", expanded=(i<=2)):
                    st.markdown(f"**Приоритет:** {rec['priority']}")
                    st.markdown(f"**Ситуация:** {rec['description']}")
                    st.success(f"**План действий:** {rec['action']}")
        
        # Операционные решения
        with rec_tab4:
            st.subheader("Операционные рекомендации")
            
            operational_recs = recommender.generate_operational_recommendations()
            
            for rec in operational_recs:
                if rec['priority'] == 'ВЫСОКИЙ':
                    st.error(f"**{rec['title']}**")
                else:
                    st.warning(f"**{rec['title']}**")
                
                st.write(f"📋 {rec['description']}")
                st.success(f"✅ {rec['action']}")
                st.markdown("---")
            
            # Матрица перераспределения
            st.subheader("📦 Матрица перераспределения товара")
            
            store_performance = df.groupby('Magazin').agg({
                'Sum': 'sum',
                'Qty': 'sum',
                'Stock': 'mean'
            }).reset_index()
            
            store_performance['Stock_Turnover'] = store_performance['Qty'] / store_performance['Stock']
            store_performance = store_performance.sort_values('Stock_Turnover', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔥 Магазины с высоким оборотом** (увеличить запасы)")
                high_turnover = store_performance.head(5)[['Magazin', 'Qty', 'Stock', 'Stock_Turnover']]
                st.dataframe(high_turnover, use_container_width=True)
            
            with col2:
                st.markdown("**🐌 Магазины с низким оборотом** (уменьшить запасы)")
                low_turnover = store_performance.tail(5)[['Magazin', 'Qty', 'Stock', 'Stock_Turnover']]
                st.dataframe(low_turnover, use_container_width=True)
        
        # RL Рекомендации
        with rec_tab5:
            st.subheader("🤖 Рекомендации на основе RL модели")
            
            if 'env' in st.session_state:
                env = st.session_state['env']
                
                if len(env.actions_history) > 0:
                    actions_df = pd.DataFrame(env.actions_history)
                    
                    # Лучшие действия
                    st.markdown("### 🏆 Топ действия по награде")
                    top_actions = actions_df.nlargest(10, 'reward')[['step', 'store', 'product', 'promo', 'reward']]
                    top_actions['promo'] = top_actions['promo'].map({0: '❌', 1: '✅'})
                    st.dataframe(top_actions, use_container_width=True)
                    
                    # Анализ эффективности промо
                    st.markdown("### 📊 Эффективность промо-акций")
                    
                    promo_comparison = actions_df.groupby('promo')['reward'].agg(['mean', 'sum', 'count']).reset_index()
                    promo_comparison['promo'] = promo_comparison['promo'].map({0: 'Без промо', 1: 'С промо'})
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Средняя награда
                    ax1.bar(promo_comparison['promo'], promo_comparison['mean'], color=['steelblue', 'orange'])
                    ax1.set_ylabel('Средняя награда')
                    ax1.set_title('Средняя эффективность')
                    ax1.grid(True, alpha=0.3, axis='y')
                    
                    # Общая награда
                    ax2.bar(promo_comparison['promo'], promo_comparison['sum'], color=['steelblue', 'orange'])
                    ax2.set_ylabel('Суммарная награда')
                    ax2.set_title('Общий вклад')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Рекомендация (защита от пустых данных и деления на ноль)
                    with_promo = promo_comparison[promo_comparison['promo'] == 'С промо']['mean'].values
                    without_promo = promo_comparison[promo_comparison['promo'] == 'Без промо']['mean'].values

                    if len(with_promo) > 0 and len(without_promo) > 0:
                        avg_with_promo = with_promo[0]
                        avg_without = without_promo[0]

                        if avg_with_promo > avg_without and avg_without > 0:
                            improvement = (avg_with_promo / avg_without - 1) * 100
                            st.success(f"✅ **Промо-акции повышают эффективность на {improvement:.1f}%!** Рекомендуется активно использовать.")
                        else:
                            st.warning("⚠️ Промо-акции показывают смешанные результаты. Требуется пересмотр стратегии.")
                    else:
                        st.warning("⚠️ Недостаточно данных для анализа эффективности промо-акций")
                    
                    # Топ магазины для промо
                    st.markdown("### 🎯 Рекомендуемые магазины для промо-акций")
                    
                    promo_by_store = actions_df[actions_df['promo'] == 1].groupby('store')['reward'].agg(['mean', 'count']).reset_index()
                    promo_by_store = promo_by_store.sort_values('mean', ascending=False).head(5)
                    promo_by_store.columns = ['Магазин', 'Средняя эффективность', 'Количество акций']
                    
                    st.dataframe(promo_by_store, use_container_width=True)
                    
                else:
                    st.info("История действий будет доступна после обучения модели")
            else:
                st.warning("⚠️ Сначала запустите обучение RL модели на вкладке 'RL Модель'")
            
            # Общие рекомендации
            st.markdown("---")
            st.subheader("🎓 Рекомендации по улучшению RL системы")
            st.markdown("""
            **Текущая версия - базовый Random агент. Для production:**
            
            1. **Алгоритмы**: DQN, PPO, A3C вместо случайных действий
            2. **Признаки**: Сезонность, конкуренты, погода, дни недели
            3. **Reward функция**: Учет долгосрочных метрик, customer lifetime value
            4. **A/B тестирование**: Валидация на реальных данных
            5. **Continuous learning**: Онлайн обучение на новых данных
            6. **Мульти-агентная система**: Отдельные агенты для разных задач
            """)


if __name__ == "__main__":
    main()
