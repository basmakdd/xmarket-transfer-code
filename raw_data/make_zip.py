"""Pack downloaded XMarket files into the zip the pipeline reads.

Put every downloaded ratings_<market>_<category>.txt.gz file in raw_data/xmarket_categories/ and run
    python raw_data/make_zip.py
This writes raw_data/xmarket_categories.zip with members named xmarket_categories/<file>."""
import glob, os, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, 'xmarket_categories', 'ratings_*.txt.gz')))
if not files:
    raise SystemExit('No ratings_*.txt.gz files found in raw_data/xmarket_categories/')
with zipfile.ZipFile(os.path.join(HERE, 'xmarket_categories.zip'), 'w', zipfile.ZIP_STORED) as z:
    for f in files:
        z.write(f, 'xmarket_categories/' + os.path.basename(f))
print(f'wrote raw_data/xmarket_categories.zip with {len(files)} files')
