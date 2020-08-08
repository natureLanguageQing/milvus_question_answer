import pandas as pd

medical_questions = pd.read_csv("medical_questions.csv", encoding="utf-8").values.tolist()
questions = []
answers = []
for i in medical_questions[:10000]:
    if i[1] not in questions and i[0] not in answers:
        if isinstance(i[1], str):
            questions.append(i[1])
        if isinstance(i[0], str):
            answers.append(i[0])
with open("medical_questions.txt", "w", encoding="utf-8") as write_file:
    for i in questions:
        i = i.replace("\n", "")
        write_file.write(i + "\n")
with open("medical_answers.txt", "w", encoding="utf-8") as write_file:
    for i in answers:
        i = i.replace("\n", "")

        write_file.write(i + "\n")
