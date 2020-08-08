import pandas as pd

medical_questions = pd.read_csv("medical_questions.csv", encoding="utf-8").dropna().drop_duplicates().values.tolist()
pd.DataFrame(medical_questions[:5000]).to_csv("medical_questions_lite.csv", encoding="utf-8", index=False,
                                              header=["answer", "question"])
