import hmac
import hashlib


# 全域哈希 hmac
def hash_with_hmac(key, data):
    key_bytes = bytes(str(key), 'utf-8')
    data_bytes = bytes(str(data), 'utf-8')
    hmac_digest = hmac.new(key_bytes, data_bytes, hashlib.sha256).hexdigest()
    return int(hmac_digest, 16)
