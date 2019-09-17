import pickle


def append_to_pickle_file(path, item):
    with open(path, 'ab') as file:
        pickle.dump(item, file)

def read_from_pickle_file(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass