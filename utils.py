
def detect_language(text):
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_count = sum(1 for char in text if 'a' <= char.lower() <= 'z')

    if chinese_count >= 1:
        return "Chinese"
    elif english_count > chinese_count:
        return "English"
