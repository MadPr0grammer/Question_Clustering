import gensim
import time

def tag_a_question(filename):
    with open(filename, encoding="utf-8") as file:
        for i, line in enumerate(file):
            '''
            Pre-process (tokenize text into individual words, remove punctuation, set to lowercase, etc)
            and then tag each question as an individual document.
            '''
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

#returns a matrix of tagged questions
def get_tagged_training_corpus(filename):
    return list(tag_a_question(filename))

#trains the model on 'training_file' and saves it in 'model_saving_file'
def train_and_save_model(training_file, model_saving_file):
    train_corpus = get_tagged_training_corpus(training_file)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    start_time = time.time()
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    end_time = time.time()
    print("Training time: ", (end_time-start_time)/60, "mins")
    model.save(model_saving_file)

def load_model_get_question_vector_matrix(model_file):
    result = []
    model = gensim.models.doc2vec.Doc2Vec.load(model_file)
    for i in range(model.corpus_count):
        result.append(model.docvecs[i])
    return result


# X = load_model_get_question_vector_matrix("question_corpus.model")
# print(len(X), len(X[0]))
#train_and_save_model("one_question_per_line.txt", "question_corpus_300.model")


#trainFile.write(','.join(map(str, a_datapoint_for_classification)))
#number of questions = 808591