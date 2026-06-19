class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))
    
    def decode(self, ids):
        return bytes(ids).decode("utf-8")

# --- test ---
text = "hello, 世界"
tokenizer = ByteTokenizer()

ids = tokenizer.encode(text)
print(ids)

decode_text = tokenizer.decode(ids)
print(decode_text)