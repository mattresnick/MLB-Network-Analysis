import unidecode
import glob

def remove_accent (feed):
    csv_f = open(feed, encoding='latin-1', mode='r')
    csv_str = csv_f.read()
    csv_str_removed_accent = unidecode.unidecode(csv_str)
    csv_f.close()
    csv_f = open(feed, 'w')
    csv_f.write(csv_str_removed_accent)
    return True

for filename in glob.glob('stats/*_stats_*.csv'):
    remove_accent(filename)
for filename in glob.glob('ranks/scaled_*_ranks_*.csv'):
    remove_accent(filename)