from self_attn_lm import LM


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()
    
    model = LM(text)
    log = model.fit()
