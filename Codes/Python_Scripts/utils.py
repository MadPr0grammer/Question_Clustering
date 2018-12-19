import pandas as pd

#This function takes the train.csv file and generates a file with one question per line (requirement for gensim)
def create_a_question_per_line_file(trainFile):
    data = pd.read_csv(trainFile, sep=',')
    with open("one_question_per_line.txt", 'w+', encoding="utf-8") as file:
        for line in data.itertuples():
            raw_question_1 = line[4]
            raw_question_2 = line[5]
            if raw_question_1:
                question_1 = str(line[4])+'\n'
            if raw_question_2:
                question_2 = str(line[5])+'\n'
            file.write(question_1)
            file.write(question_2)

# Writes a matrix to a file as csv
def write_matrix_to_file(matrix, saving_filename):
    with open(saving_filename, "w+") as file:
        for row in matrix:
            file.write(','.join(map(str, row)))
            file.write("\n")

def write_1d_matrix_to_file(matrix, saving_filename):
    with open(saving_filename, "w+") as file:
        for row in matrix:
            file.write(str(row))
            file.write("\n")

#create_a_question_per_line_file("train.csv")
#header = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
# data = pd.read_csv("train.csv", sep=',')
# print(data.head())