text = []


def getText(role,content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    if getlength(text) + len(content) > 8000:
        checklen(text)
    text.append(jsoncon)
    return text


def getlength(text):
    length = sum(len(item["content"]) for item in text)
    return length


def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text