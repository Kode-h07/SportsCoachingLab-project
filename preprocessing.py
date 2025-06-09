import pandas as pd
from googletrans import Translator

# Load the data
df = pd.read_csv("cleaned_output.csv")

# Drop the specified columns
df = df.drop(
    columns=[
        "타임스탬프",
        "농구하러 올떄 사용하는 교통수단",
        "가장 자주 사용하는 교통수단",
        "현재 거주 중인 지역(동까지 자세히 입력 부탁드립니다) (예시: 서울시 송파구 잠실동)",
        "운동삼아 이동할 수 있는 최대 거리(소요시간 기준)",
    ],
    errors="ignore",
)


# Show result
df.to_csv("cleaned_output.csv", index=False, encoding="utf-8-sig")

df = pd.read_csv("pretranslated.csv")
df.drop(columns=["address"], inplace=True)
df["성별"] = df["성별"].apply(lambda x: 1 if x.strip() == "여성" else 0)
df["직업 종류"] = df["직업 종류"].apply(lambda x: 1 if x.strip() == "직장인" else 0)
df.to_csv("preprocessed.csv", index=False)
