import urllib
import re

from pathlib import Path


def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def clean_url(url):
    url = Path(url).as_posix().replace(":/", "://")
    return urllib.parse.unquote(url).split("?")[0]


def url2file(url):
    return Path(clean_url(url)).name
