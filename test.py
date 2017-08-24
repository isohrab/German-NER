with open("data/words.txt",encoding='utf-8') as f:
    maxx = 0
    sum = 0
    counter = 0
    for line in f:
        word = line.strip()
        lens = len(word)
        sum += lens
        counter += 1
        if lens > maxx:
            maxx = lens

    print("sum:", sum/counter, "max:", maxx)
