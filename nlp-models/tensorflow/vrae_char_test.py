from vrae_char import VRAE


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()
    
    model = VRAE(text)
    log = model.fit()
