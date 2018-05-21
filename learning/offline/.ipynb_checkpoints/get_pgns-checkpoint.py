import requests
from bs4 import BeautifulSoup
import os
import re
import sys

def arg_checks(args):
    if len(args) < 3:
        print('Note enough arguments provided.')
        print('Run file in the following way: ')
        print('> python get_pgn.py <Number of PGNs to Scrape> <Output Directory>')
        sys.exit()
    elif len(args)>3:
        print('Too many arguments provided')
        print('Run file in the following way: ')
        print('> python get_pgn.py <Number of PGNs to Scrape> <Output Directory>')
        sys.exit()


def get_html(url):
    # Grab file links
    html = BeautifulSoup(requests.get(url).content, 'lxml')
    data_links = html.text.split('\n')
    return data_links


def download(list_of_links, number_of_files, output_directory):
    for link in list_of_links[-number_of_files:]:
        datetime = re.search('[0-9]{4}\-[0-9]{2}', link).group().replace('-', '_')
        filename = '{}/{}.pgn.bz2'.format(output_directory, datetime)
        os.popen("curl '{}' > {}".format(link, filename)).read()
        print('Uncompressing...')
        if os.listdir(output_directory):
            overwrite = input(
                'Some .pgn files already exist, if duplicates exist, shall they be overwritten (y/n)? ')
            if overwrite == 'y':
                os.popen('bzip2 -d -f {}'.format(filename)).read()
        else:
            os.popen('bzip2 -d {}'.format(filename)).read()
        print('-' * 80)

# Output directory
directory = 'data'


if __name__=='__main__':
    arg_checks(sys.argv)
    links = get_html('https://database.lichess.org/standard/list.txt')
    print('Grabbing and extracting {} .pgn files.'.format(sys.argv[1]))
    download(links, int(sys.argv[1]), sys.argv[2])
    print('{} pgn successfully files created'.format(len(os.listdir(directory))))

