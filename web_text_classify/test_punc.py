# %%

def remove_special_characters(text):
    tokens = list(text) #tokens为分词后的文本
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) #正则匹配特殊符号
    print(pattern)
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
a = "\ue123helloworld"
tmp1 = remove_special_characters(a)
print(tmp1)
