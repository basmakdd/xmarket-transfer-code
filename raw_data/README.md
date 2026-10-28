# Raw data (not included)

The study uses the XMarket dataset (Bonab et al., CIKM 2021), <https://xmrec.github.io/>.
The raw files are not redistributed here; download them yourself and cite the dataset paper.

1. From the per-market pages (e.g. <https://xmrec.github.io/data/de>) download the category *ratings* files
   `ratings_<market>_<category>.txt.gz` for the 7 markets `de jp in fr ca mx uk` and these 16 categories:
   Electronics, Home_and_Kitchen, Cell_Phones_and_Accessories, Movies_and_TV, Sports_and_Outdoors, Toys_and_Games,
   Office_Products, CDs_and_Vinyl, Musical_Instruments, Books, Arts_Crafts_and_Sewing, Automotive, Digital_Music,
   Grocery_and_Gourmet_Food, Industrial_and_Scientific, Kindle_Store.
   Not every market has every category; missing files are skipped automatically (98 files exist).
2. Put them in `raw_data/xmarket_categories/` and run `python raw_data/make_zip.py`.
   Alternatively point the `XMARKET_ZIP` environment variable at an existing zip whose members are
   `xmarket_categories/ratings_<market>_<category>.txt.gz`.
