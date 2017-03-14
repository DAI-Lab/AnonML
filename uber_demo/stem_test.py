import io
import pycurl as curl
from stem import process
from stem.util import term
PORT = 7000

def query(url):
    out = io.BytesIO()
    query = curl.Curl()
    query.setopt(curl.URL, url)
    query.setopt(curl.PROXY, 'localhost')
    query.setopt(curl.PROXYPORT, PORT)
    query.setopt(curl.PROXYTYPE, curl.PROXYTYPE_SOCKS5_HOSTNAME)
    query.setopt(curl.WRITEFUNCTION, out.write)

    try:
        query.perform()
        return out.getvalue()
    except curl.error as e:
        return 'Unable to reach %s (%s)' % (url, e)

def print_bootstrap_lines(line):
    if 'Bootastrapped ' in line:
        print term.format(line, term.Color.BLUE)

if __name__ == '__main__':
    print term.format('Starting Tor:\n', term.Attr.BOLD)

    tor_ps = process.launch_tor_with_config(
        config = {
            'SocksPort': str(PORT),
            'ExitNodes': '{ru}',
        },
        init_msg_handler = print_bootstrap_lines,
    )
    print term.format('Checking our endpoint:\n', term.Attr.BOLD)
    print term.format(query('https://cyphe.rs/'), term.Color.BLUE)

    tor_ps.kill()
