import requests 

class VNLPClient():
  """
    >>> vc = VNLPClient("http://localhost:39000")
    >>> vc.tokenize("Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây.")
  """
  def __init__(self, url):
    self.url = url
    self.url_tokenize = f"{url}/tokenize"
    self.url_postag = f"{url}/postag"
    self.url_ner = f"{url}/ner"
    self.url_depparse = f"{url}/depparse"
    
  def _req(self, url, text):
    rv = requests.post(url, json={"text": text})
    if (rv.status_code != 200):
      return rv.content
    return rv.json()
    
  def tokenize(self, text):
    return self._req(self.url_tokenize, text)
    
  def postag(self, text):
    return self._req(self.url_postag, text)
    
  def ner(self, text):
    return self._req(self.url_ner, text)
    
  def depparse(self, text):
    return self._req(self.url_depparse, text)

