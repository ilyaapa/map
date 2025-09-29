import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import hashlib

app = FastAPI()

# Константы для разбиения на группы
SALT = "my_salt_value"
GROUP_SPLIT_RATIO = 0.5  # 50/50 split

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

engine = create_engine(
    # настройки удалены
)

# Загрузка таблицы по частям
def batch_load_sql(query: str):
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f'Got chunk: {len(chunk_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# Группа юзера
def get_exp_group(user_id: int) -> str:
    hash_str = hashlib.md5((str(user_id) + SALT).encode('utf-8')).hexdigest()
    hash_int = int(hash_str, 16)
    return "test" if (hash_int % 100) < (GROUP_SPLIT_RATIO * 100) else "control"

# Путь до модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        if path.endswith('_control'):
            return '/workdir/user_input/model_control'
        elif path.endswith('_test'):
            return '/workdir/user_input/model_test'
        else:
            return '/workdir/user_input/model'  # fallback на случай, если путь не совпал
    else:
        return path

# Загрузка постов с лайками
def load_liked_posts():
    logger.info('loading liked posts')
    engine.dispose()
    return batch_load_sql("""
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action = 'like'""")

# Загрузка контрольной модели
def load_control_model():
    model_path = get_model_path("C:/Users/model_control")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# Загрузка тестовой модели
def load_test_model():
    model_path = get_model_path("C:/Users/model_test")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# Загрузка фичей для контрольной группы
def load_control_features(liked_posts):
    logger.info('loading posts features')
    engine.dispose()
    posts_features = pd.read_sql(
        'SELECT * FROM public.l_22_posts_features',
        con=engine
    )

    logger.info('loading users features')
    engine.dispose()
    users_features = pd.read_sql(
        'SELECT * FROM public.l_22_users_features',
        con=engine
    )

    return liked_posts, posts_features, users_features

# Загрузка фичей для тестовой группы
def load_test_features(liked_posts):
    logger.info('loading posts features')
    engine.dispose()
    posts_features = pd.read_sql(
        'SELECT * FROM public.m_3_posts_features_v7',
        con=engine
    )

    logger.info('loading users features')
    engine.dispose()
    users_features = pd.read_sql(
        'SELECT * FROM public.m_3_users_features_v7',
        con=engine
    )

    return liked_posts, posts_features, users_features

# Инициализация при старте
logger.info('loading control model')
model_control = load_control_model()
logger.info('loading test model')
model_test = load_test_model()

# Загружаем liked_posts один раз
liked_posts = load_liked_posts()

logger.info('loading control features')
control_features = load_control_features(liked_posts)
logger.info('loading test features')
test_features = load_test_features(liked_posts)
logger.info('service is up and running')

# Построение рекомендаций для контрольной группы
def get_control_recommendations(id: int, time: datetime, limit: int):
    liked_posts, posts_features, users_features = control_features

    logger.info(f'control group user_id: {id}')
    user_features = users_features.loc[users_features.user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    posts_data = posts_features.drop(['text', 'topic'], axis=1)
    content = posts_features[['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_data.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['month'] = time.month
    user_posts_features['hour'] = time.hour
    user_posts_features['weekday'] = time.weekday()

    pool = Pool(user_posts_features, cat_features=['city'])
    predicts = model_control.predict_proba(pool)[:, 1]
    user_posts_features['predicts'] = predicts

    liked = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_posts = user_posts_features[~user_posts_features.index.isin(liked)]

    recommended_posts = filtered_posts.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            'id': i,
            'text': content[content.post_id == i].text.values[0],
            'topic': content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]

# Построение рекомендаций для тестовой группы
def get_test_recommendations(id: int, time: datetime, limit: int):
    liked_posts, posts_features, users_features = test_features

    logger.info(f'test group user_id: {id}')
    user_features = users_features.loc[users_features.user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    posts_data = posts_features.drop(['text'], axis=1)
    content = posts_features[['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_data.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['month'] = time.month
    user_posts_features['hour'] = time.hour

    object_cols = ['topic', 'TextCluster', 'gender', 'country',
                 'city', 'exp_group', 'hour', 'month', 'os', 'source']
    pool = Pool(user_posts_features, cat_features=object_cols)
    predicts = model_test.predict_proba(pool)[:, 1]
    user_posts_features['predicts'] = predicts

    liked = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_posts = user_posts_features[~user_posts_features.index.isin(liked)]

    recommended_posts = filtered_posts.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            'id': i,
            'text': content[content.post_id == i].text.values[0],
            'topic': content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]

# endpoint
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    exp_group = get_exp_group(id)
    logger.info(f'user {id} assigned to group {exp_group}')

    if exp_group == 'control':
        recommendations = get_control_recommendations(id, time, limit)
    elif exp_group == 'test':
        recommendations = get_test_recommendations(id, time, limit)
    else:
        raise ValueError('unknown group')

    return Response(
        exp_group=exp_group,
        recommendations=recommendations
    )
