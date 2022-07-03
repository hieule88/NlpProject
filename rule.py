import re
import ipaddress

def filterEmail(doc):
    email = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(email, doc):
        return True
    else:
        return False

def filterUrl(doc):
    url = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    if re.search(url, doc):
        return True 
    else:
        return False

def filterIP(doc):
    try: 
        ipaddress.ip_address(doc)
        return True
    except:
        return False

print(filterIP("2001:0db8:85a3:0000:0000:8a2e:0370:7334"))
