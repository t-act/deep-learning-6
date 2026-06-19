text = "hello, 世界"
# print(list(text))

# print(ord("h"))
# print(chr(104))

class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join([chr(i) for i in ids])

tokenizer = CharTokenizer()
text = "hello, 世界"

ids = tokenizer.encode(text)
print(ids)
decode_text = tokenizer.decode(ids)
print(decode_text)