# COVID19 Text Analysis

COVID-19 infections summaries tf-idf text analysis, clustering and labeling countries by keywords.

## About the project

### General stats of top 10 infected countries

For each country compute number of infections, average infections age and female percentage out of total infections.
Results of top 10 infected countries are written to country_stats.txt, sorted by number of infections:

```
       country  infected_count  average_age  female_percent
0        China             197    49.017187       37.055838
1        Japan             190    55.336499       33.684211
2  South Korea             114    47.996852       42.982456
3    Hong Kong              94    56.005146       50.000000
4    Singapore              93    43.709151       39.784946
5      Germany              54    47.080511       12.962963
6     Thailand              41    48.905484       26.829268
7       France              39    48.157884       17.948718
8        Spain              34    45.979058       20.588235
9       Taiwan              34    51.719149       52.941176
```

### Labeling countries by summaries keywords

Each summary is represented by a [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vector, and labeled by it's top 5 tokens with highest tf-idf value. Then, for each country we create a list of it's summaries labels, and again by using tf-idf technique each country is labeled by it's top 15 token, w.r.t their tf-idf values.
Results are written to country_lables.txt (partial list):

```
Hong Kong:
['onn', 'maternal', 'fractures', 'mouth', 'hours', 'coworker', 'cousin', 'others', 'mainland', 'via', 'train', 'forth', 'grandmother', 'fall', 'building']

France:
['confinement', 'woman', 'solitary', 'journey', 'origin', 'elderly', 'date', 'returning', 'spent', 'plane', 'friend', 'positive', 'cruise', 'person', 'visiting']

Spain:
['nurse', 'watch', 'northern', 'previous', 'sports', 'coronavirus', 'flulike', 'holidaying', 'islands', 'sportswriter', 'watched', 'game', 'car', 'studying', 'time']
```

Notice that some interesting trends can be deduced from these results. In Spain, for instance, some of the labels are related to the field of sports (sports, sportswriter, game) and in Hong Kong it seems like some of the labels are family members (cousin, grandmother).
In France, some labels are related to traveling (journey, plane, cruise).

It is likely to assume that lots of other labels, and even the labels mentioned above are irrelevant due to relatively small amount of data, but it might be interesting using this tf-idf analysis method on a larger COVID-19 dataset, and performe some adjustments if needed.

### Summaries clustering

TODO.

### Labeling clusters by summaries keywords

TODO.

## Getting Started (for Windows or Linux)

### Installing

Running these commands will get you a copy of the project on your local machine, as well as install all relevant libraries:

```
git clone https://github.com/omermadmon/prj_COVID19.git
cd prj_COVID19
pip install -r requirements.txt
```

### Running

```
python main.py
```

Results will be written to 4 txt files:
* country_stats.txt
* country_lables.txt
* summaries_clusters.txt
* cluster_lables.txt

## Credits

* [COVID-19 dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
* [tf-idf implementation](https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76)
