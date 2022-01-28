# NOTES

## API_KEY

- id : 1
- domain : <http://unc.nc>
- name : _chemo-diversity_
- key : _7047b3a8cf46d922d5d5ca71ff531b7d_

## API

### ScienceDirect

Voir <https://github.com/ElsevierDev/elsapy>

- <https://api.elsevier.com/content/search/sciencedirect>
- <https://datasearch.elsevier.com/api/docs>
- <https://dev.elsevier.com/documentation/ScopusSearchAPI.wadl>
- <https://dev.elsevier.com/tips/ScopusSearchTips.htm>

```bash
http --print HBhb PUT https://api.elsevier.com/content/search/sciencedirect X-ELS-APIKey:7047b3a8cf46d922d5d5ca71ff531b7d < search.json
```

### Scopus

Recherche manuelle <https://www.scopus.com/results/results.uri?sid=aba53c45c2b93010ce7dbdd6dcc88e07&src=s&sot=b&sdt=b&origin=searchbasic&rr=&sl=31&s=(KEY(alkaloid)%20AND%20KEY(cancer))&searchterm1=alkaloid&onnector=AND&searchterm2=cancer&searchTerms=&connectors=&field1=KEY&field2=KEY&fields=>

```bash
http GET "https://api.elsevier.com/content/search/scopus?query=KEY(alkaloid)%20AND%20KEY(cancer)&count=1" X-ELS-APIKey:7047b3a8cf46d922d5d5ca71ff531b7d
```

## Résultat HTTP

```raw
GET /content/search/scopus?query=KEY(alkaloid)%20AND%20KEY(cancer)&count=1 HTTP/1.1
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
Host: api.elsevier.com
User-Agent: HTTPie/1.0.3
X-ELS-APIKey: 7047b3a8cf46d922d5d5ca71ff531b7d



HTTP/1.1 200 OK
CF-Cache-Status: DYNAMIC
CF-RAY: 688c82ea7d6dd4d5-NOU
Connection: keep-alive
Content-Encoding: gzip
Content-Type: application/json;charset=UTF-8
Date: Fri, 03 Sep 2021 05:17:34 GMT
Expect-CT: max-age=604800, report-uri="https://report-uri.cloudflare.com/cdn-cgi/beacon/expect-ct"
Server: cloudflare
Transfer-Encoding: chunked
Vary: Accept-Encoding
Vary: Origin
Vary: Access-Control-Request-Method
Vary: Access-Control-Request-Headers
X-ELS-APIKey: 7047b3a8cf46d922d5d5ca71ff531b7d
X-ELS-ReqId: 54f5da454b79f8fd
X-ELS-ResourceVersion: default
X-ELS-Status: OK
X-ELS-TransId: 64531c1fa0f70ba1
X-RateLimit-Limit: 20000
X-RateLimit-Remaining: 19996
X-RateLimit-Reset: 1631249428000
allow: GET
```

NB : epoch 1631249428000 = Friday, 10 September 2021 15:50:28 GMT+11:00

### Limits

```raw
X-RateLimit-Limit       <----Shows API quota setting
X-RateLimit-Remaining   <----Shows API remaining quota
X-RateLimit-Reset       1234567891 <----Date/Time in Epoch seconds when API quota resets
```

For _Scopus Search_, see <https://dev.elsevier.com/api_key_settings.html>

- Weekly Quota : 20.000
- Requests/second : 9

## Examples

### `results.json`

```json
{
  "search-results": {
    "opensearch:totalResults": "7794",
    "opensearch:startIndex": "0",
    "opensearch:itemsPerPage": "1",
    "opensearch:Query": { "@role": "request", "@searchTerms": "KEY(alkaloid) AND KEY(cancer)", "@startPage": "0" },
    "link": [
      {
        "@_fa": "true",
        "@ref": "self",
        "@href": "https://api.elsevier.com/content/search/scopus?start=0&count=1&query=KEY%28alkaloid%29+AND+KEY%28cancer%29",
        "@type": "application/json"
      },
      {
        "@_fa": "true",
        "@ref": "first",
        "@href": "https://api.elsevier.com/content/search/scopus?start=0&count=1&query=KEY%28alkaloid%29+AND+KEY%28cancer%29",
        "@type": "application/json"
      },
      {
        "@_fa": "true",
        "@ref": "next",
        "@href": "https://api.elsevier.com/content/search/scopus?start=1&count=1&query=KEY%28alkaloid%29+AND+KEY%28cancer%29",
        "@type": "application/json"
      },
      {
        "@_fa": "true",
        "@ref": "last",
        "@href": "https://api.elsevier.com/content/search/scopus?start=4999&count=1&query=KEY%28alkaloid%29+AND+KEY%28cancer%29",
        "@type": "application/json"
      }
    ],
    "entry": [
      {
        "@_fa": "true",
        "link": [
          {
            "@_fa": "true",
            "@ref": "self",
            "@href": "https://api.elsevier.com/content/abstract/scopus_id/85112205136"
          },
          {
            "@_fa": "true",
            "@ref": "author-affiliation",
            "@href": "https://api.elsevier.com/content/abstract/scopus_id/85112205136?field=author,affiliation"
          },
          {
            "@_fa": "true",
            "@ref": "scopus",
            "@href": "https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=85112205136&origin=inward"
          },
          {
            "@_fa": "true",
            "@ref": "scopus-citedby",
            "@href": "https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=85112205136&origin=inward"
          },
          {
            "@_fa": "true",
            "@ref": "full-text",
            "@href": "https://api.elsevier.com/content/article/eid/1-s2.0-S0304419X21001049"
          }
        ],
        "prism:url": "https://api.elsevier.com/content/abstract/scopus_id/85112205136",
        "dc:identifier": "SCOPUS_ID:85112205136",
        "eid": "2-s2.0-85112205136",
        "dc:title": "βIII-tubulin overexpression in cancer: Causes, consequences, and potential therapies",
        "dc:creator": "Kanakkanthara A.",
        "prism:publicationName": "Biochimica et Biophysica Acta - Reviews on Cancer",
        "prism:issn": "0304419X",
        "prism:eIssn": "18792561",
        "prism:volume": "1876",
        "prism:issueIdentifier": "2",
        "prism:pageRange": null,
        "prism:coverDate": "2021-12-01",
        "prism:coverDisplayDate": "December 2021",
        "prism:doi": "10.1016/j.bbcan.2021.188607",
        "pii": "S0304419X21001049",
        "citedby-count": "0",
        "affiliation": [
          {
            "@_fa": "true",
            "affilname": "Mayo Clinic",
            "affiliation-city": "Rochester",
            "affiliation-country": "United States"
          }
        ],
        "prism:aggregationType": "Journal",
        "subtype": "re",
        "subtypeDescription": "Review",
        "article-number": "188607",
        "source-id": "80280",
        "openaccess": "0",
        "openaccessFlag": false
      }
    ]
  }
}
```

### `search.json`

```json
{
  "title": "Articulation Disorders",
  "filters": {
    "openAccess": true
  },
  "loadedAfter": "2018-06-01T00:00:00Z"
}
```

## Extra on pandas

```python
# no names attribute to ensure read -> write is identity
, names = ["a-class", "compound"]
, names = ["a-class", "activity"]

# remove first level on index
df = df.droplevel("C-Class", axis = 0)
df = df.droplevel("A-Class", axis = 1)

compounds = {y: x for (x, y) in df.index}
activities = {y: x for (x, y) in df.columns}

compounds_classes, compounds = df.index.levels
activities_classes, activities = df.columns.levels
```

## Complete query "all compounds"

```raw
((KEY("acridine")) OR (KEY("benzylamine")) OR (KEY("colchicine")) OR (KEY("cyclopeptide")) OR (KEY("imidazole")) OR (KEY("indole")) OR (KEY("indolizidine")) OR (KEY("isoquinoline")) OR (KEY("isoxazole")) OR (KEY("muscarine")) OR (KEY("oxazole")) OR (KEY("phenylethylamine")) OR (KEY("piperidine")) OR (KEY("purine")) OR (KEY("putrescine")) OR (KEY("pyridine")) OR (KEY("pyrrolidine")) OR (KEY("pyrrolizidine")) OR (KEY("quinazoline")) OR (KEY("quinoline")) OR (KEY("quinolizidine")) OR (KEY("spermidine")) OR (KEY("spermine")) OR (KEY("thiazole")) OR (KEY("tropane")) OR (KEY("acetophenone")) OR (KEY("anthraquinone")) OR (KEY("biflavonoids")) OR (KEY("flavonoids")) OR (KEY("isoflavanoids")) OR (KEY("lignans")) OR (KEY("naphthoquinone")) OR (KEY("phenol")) OR (KEY("phenolic acid")) OR (KEY("phenylpropanoid")) OR (KEY("stilbene")) OR (KEY("tannin")) OR (KEY("xanthone")) OR (KEY("acetogenin")) OR (KEY("ansamycin")) OR (KEY("macrolide")) OR (KEY("polyene")) OR (KEY("polyether")) OR (KEY("tetracycline")) OR (KEY("diterpene")) OR (KEY("hemiterpene")) OR (KEY("monoterpene")) OR (KEY("norisoprenoid")) OR (KEY("polyterpene")) OR (KEY("sesquiterpene")) OR (KEY("sesterterpene")) OR (KEY("tetraterpene") OR KEY("carotenoid") OR KEY("xanthophyll")) OR (KEY("triterpene")))
```

1,138,377 document results

## All activities

```raw
((KEY("antioxidant")) OR (KEY("drought")) OR (KEY("metal")) OR (KEY("uv")) OR (KEY("salt")) OR (KEY("antifeedant")) OR (KEY("arbuscula")) OR (KEY("attractant")) OR (KEY("germination")) OR (KEY("herbicidal")) OR (KEY("hppd")) OR (KEY("hyphal")) OR (KEY("phytotoxicity")) OR (KEY("quorum sensing")) OR (KEY("repulsive")) OR (KEY("toxicant")) OR (KEY("antidiabetic")) OR (KEY("cardiovascular")) OR (KEY("obesity")) OR (KEY("rheumatism")) OR (KEY("antibacterial")) OR (KEY("antifungal")) OR (KEY("antimicrobial")) OR (KEY("antiparasitic")) OR (KEY("antiviral")) OR (KEY("anti-inflammatory")) OR (KEY("arthritis")) OR (KEY("burns")) OR (KEY("wound")) OR (KEY("anticancer")) OR (KEY("cytotoxicity")) OR (KEY("sedative")) OR (KEY("toxicity")))
```

5,341,029 document results

## Full domain

```raw
((KEY("acridine")) OR (KEY("benzylamine")) OR (KEY("colchicine")) OR (KEY("cyclopeptide")) OR (KEY("imidazole")) OR (KEY("indole")) OR (KEY("indolizidine")) OR (KEY("isoquinoline")) OR (KEY("isoxazole")) OR (KEY("muscarine")) OR (KEY("oxazole")) OR (KEY("phenylethylamine")) OR (KEY("piperidine")) OR (KEY("purine")) OR (KEY("putrescine")) OR (KEY("pyridine")) OR (KEY("pyrrolidine")) OR (KEY("pyrrolizidine")) OR (KEY("quinazoline")) OR (KEY("quinoline")) OR (KEY("quinolizidine")) OR (KEY("spermidine")) OR (KEY("spermine")) OR (KEY("thiazole")) OR (KEY("tropane")) OR (KEY("acetophenone")) OR (KEY("anthraquinone")) OR (KEY("biflavonoids")) OR (KEY("flavonoids")) OR (KEY("isoflavanoids")) OR (KEY("lignans")) OR (KEY("naphthoquinone")) OR (KEY("phenol")) OR (KEY("phenolic acid")) OR (KEY("phenylpropanoid")) OR (KEY("stilbene")) OR (KEY("tannin")) OR (KEY("xanthone")) OR (KEY("acetogenin")) OR (KEY("ansamycin")) OR (KEY("macrolide")) OR (KEY("polyene")) OR (KEY("polyether")) OR (KEY("tetracycline")) OR (KEY("diterpene")) OR (KEY("hemiterpene")) OR (KEY("monoterpene")) OR (KEY("norisoprenoid")) OR (KEY("polyterpene")) OR (KEY("sesquiterpene")) OR (KEY("sesterterpene")) OR (KEY("tetraterpene") OR KEY("carotenoid") OR KEY("xanthophyll")) OR (KEY("triterpene"))) AND ((KEY("antioxidant")) OR (KEY("drought")) OR (KEY("metal")) OR (KEY("uv")) OR (KEY("salt")) OR (KEY("antifeedant")) OR (KEY("arbuscula")) OR (KEY("attractant")) OR (KEY("germination")) OR (KEY("herbicidal")) OR (KEY("hppd")) OR (KEY("hyphal")) OR (KEY("phytotoxicity")) OR (KEY("quorum sensing")) OR (KEY("repulsive")) OR (KEY("toxicant")) OR (KEY("antidiabetic")) OR (KEY("cardiovascular")) OR (KEY("obesity")) OR (KEY("rheumatism")) OR (KEY("antibacterial")) OR (KEY("antifungal")) OR (KEY("antimicrobial")) OR (KEY("antiparasitic")) OR (KEY("antiviral")) OR (KEY("anti-inflammatory")) OR (KEY("arthritis")) OR (KEY("burns")) OR (KEY("wound")) OR (KEY("anticancer")) OR (KEY("cytotoxicity")) OR (KEY("sedative")) OR (KEY("toxicity")))
```

292,709 document results

## Output for test dataset

```raw
INFO:chemo-diversity-scopus:output dir is '/home/romulus/Documents/Rec-DM-chemotaxonomy/scopus/results'
INFO:chemo-diversity-scopus:Scopus API key {'X-ELS-APIKey': '74c71d241426db269fba091507dfde38'}
DEBUG:chemo-diversity-scopus:dataset data/tests.csv read
INFO:chemo-diversity-scopus:2 compounds (with 1 classes)
INFO:chemo-diversity-scopus:2 activities (with 1 classes)
DEBUG:chemo-diversity-scopus:all compounds ['sociology', 'linguistics']
DEBUG:chemo-diversity-scopus:all activities ['databases', 'web']
INFO:chemo-diversity-scopus:total number of queries: 25
INFO:chemo-diversity-scopus:query is (['sociology', 'databases'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19966
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=165 in 1.459132 sec
INFO:chemo-diversity-scopus:query is (['databases'], ['sociology'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19965
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=2153 in 0.684227 sec
INFO:chemo-diversity-scopus:query is (['sociology'], ['databases'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19964
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=128 in 1.320619 sec
INFO:chemo-diversity-scopus:query is ([], ['sociology', 'databases'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19963
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=977 in 1.199515 sec
INFO:chemo-diversity-scopus:query is (['sociology', 'web'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19962
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=141 in 0.620706 sec
INFO:chemo-diversity-scopus:query is (['web'], ['sociology'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19961
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=1090 in 1.334602 sec
INFO:chemo-diversity-scopus:query is (['sociology'], ['web'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19960
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=152 in 0.669188 sec
INFO:chemo-diversity-scopus:query is ([], ['sociology', 'web'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19959
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=2040 in 1.267862 sec
INFO:chemo-diversity-scopus:query is (['linguistics', 'databases'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19958
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=2155 in 1.311808 sec
INFO:chemo-diversity-scopus:query is (['databases'], ['linguistics'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19957
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=163 in 1.189209 sec
INFO:chemo-diversity-scopus:query is (['linguistics'], ['databases'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19956
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=979 in 0.512150 sec
INFO:chemo-diversity-scopus:query is ([], ['linguistics', 'databases'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19955
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=126 in 0.696818 sec
INFO:chemo-diversity-scopus:query is (['linguistics', 'web'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19954
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=1092 in 1.187027 sec
INFO:chemo-diversity-scopus:query is (['web'], ['linguistics'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19953
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=139 in 0.456412 sec
INFO:chemo-diversity-scopus:query is (['linguistics'], ['web'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19952
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=2042 in 0.515974 sec
INFO:chemo-diversity-scopus:query is ([], ['linguistics', 'web'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19951
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=150 in 1.266023 sec
INFO:chemo-diversity-scopus:query is (['sociology'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19950
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=293 in 0.529831 sec
INFO:chemo-diversity-scopus:query is ([], ['sociology'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19949
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=3130 in 1.448143 sec
INFO:chemo-diversity-scopus:query is (['linguistics'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19948
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=3134 in 1.287377 sec
INFO:chemo-diversity-scopus:query is ([], ['linguistics'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19947
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=289 in 1.214314 sec
INFO:chemo-diversity-scopus:query is (['databases'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19946
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=2318 in 1.280899 sec
INFO:chemo-diversity-scopus:query is ([], ['databases'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19945
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=1105 in 1.225449 sec
INFO:chemo-diversity-scopus:query is (['web'], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19944
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=1231 in 1.324022 sec
INFO:chemo-diversity-scopus:query is ([], ['web'])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19943
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=2192 in 0.503255 sec
INFO:chemo-diversity-scopus:query is ([], [])
DEBUG:chemo-diversity-scopus:X-RateLimit-Remaining=19942
DEBUG:chemo-diversity-scopus:query_scopus(): results_nb=3423 in 0.563425 sec
```

### Another json

```json
{
  "search-results": {
    "opensearch:totalResults": "1225",
    "opensearch:startIndex": "0",
    "opensearch:itemsPerPage": "1",
    "opensearch:Query": {
      "@role": "request",
      "@searchTerms": "( ((KEY(acridine)) OR (KEY(anthraquinone)) OR (KEY(tetraterpene) OR KEY(carotenoid) OR KEY(xanthophyll)) OR (KEY(triterpene))) AND ((KEY(germination)) OR (KEY(herbicidal)) OR (KEY(cytotoxicity)) OR (KEY(sedative))) AND ((KEY(acridine))) AND NOT ((KEY(cytotoxicity) AND KEY(toxicity))) ) AND ( DOCTYPE( \"ar\" ) )",
      "@startPage": "0"
    },
    "link": [
      {
        "@_fa": "true",
        "@ref": "self",
        "@href": "https://api.elsevier.com/content/search/scopus?start=0&count=1&query=%28+%28%28KEY%28acridine%29%29+OR+%28KEY%28anthraquinone%29%29+OR+%28KEY%28tetraterpene%29+OR+KEY%28carotenoid%29+OR+KEY%28xanthophyll%29%29+OR+%28KEY%28triterpene%29%29%29+AND+%28%28KEY%28germination%29%29+OR+%28KEY%28herbicidal%29%29+OR+%28KEY%28cytotoxicity%29%29+OR+%28KEY%28sedative%29%29%29+AND+%28%28KEY%28acridine%29%29%29+AND+NOT+%28%28KEY%28cytotoxicity%29+AND+KEY%28toxicity%29%29%29+%29+AND+%28+DOCTYPE%28+%22ar%22+%29+%29",
        "@type": "application/json"
      },
      {
        "@_fa": "true",
        "@ref": "first",
        "@href": "https://api.elsevier.com/content/search/scopus?start=0&count=1&query=%28+%28%28KEY%28acridine%29%29+OR+%28KEY%28anthraquinone%29%29+OR+%28KEY%28tetraterpene%29+OR+KEY%28carotenoid%29+OR+KEY%28xanthophyll%29%29+OR+%28KEY%28triterpene%29%29%29+AND+%28%28KEY%28germination%29%29+OR+%28KEY%28herbicidal%29%29+OR+%28KEY%28cytotoxicity%29%29+OR+%28KEY%28sedative%29%29%29+AND+%28%28KEY%28acridine%29%29%29+AND+NOT+%28%28KEY%28cytotoxicity%29+AND+KEY%28toxicity%29%29%29+%29+AND+%28+DOCTYPE%28+%22ar%22+%29+%29",
        "@type": "application/json"
      },
      {
        "@_fa": "true",
        "@ref": "next",
        "@href": "https://api.elsevier.com/content/search/scopus?start=1&count=1&query=%28+%28%28KEY%28acridine%29%29+OR+%28KEY%28anthraquinone%29%29+OR+%28KEY%28tetraterpene%29+OR+KEY%28carotenoid%29+OR+KEY%28xanthophyll%29%29+OR+%28KEY%28triterpene%29%29%29+AND+%28%28KEY%28germination%29%29+OR+%28KEY%28herbicidal%29%29+OR+%28KEY%28cytotoxicity%29%29+OR+%28KEY%28sedative%29%29%29+AND+%28%28KEY%28acridine%29%29%29+AND+NOT+%28%28KEY%28cytotoxicity%29+AND+KEY%28toxicity%29%29%29+%29+AND+%28+DOCTYPE%28+%22ar%22+%29+%29",
        "@type": "application/json"
      },
      {
        "@_fa": "true",
        "@ref": "last",
        "@href": "https://api.elsevier.com/content/search/scopus?start=1224&count=1&query=%28+%28%28KEY%28acridine%29%29+OR+%28KEY%28anthraquinone%29%29+OR+%28KEY%28tetraterpene%29+OR+KEY%28carotenoid%29+OR+KEY%28xanthophyll%29%29+OR+%28KEY%28triterpene%29%29%29+AND+%28%28KEY%28germination%29%29+OR+%28KEY%28herbicidal%29%29+OR+%28KEY%28cytotoxicity%29%29+OR+%28KEY%28sedative%29%29%29+AND+%28%28KEY%28acridine%29%29%29+AND+NOT+%28%28KEY%28cytotoxicity%29+AND+KEY%28toxicity%29%29%29+%29+AND+%28+DOCTYPE%28+%22ar%22+%29+%29",
        "@type": "application/json"
      }
    ],
    "entry": [
      {
        "@_fa": "true",
        "link": [
          {
            "@_fa": "true",
            "@ref": "self",
            "@href": "https://api.elsevier.com/content/abstract/scopus_id/85122349027"
          },
          {
            "@_fa": "true",
            "@ref": "author-affiliation",
            "@href": "https://api.elsevier.com/content/abstract/scopus_id/85122349027?field=author,affiliation"
          },
          {
            "@_fa": "true",
            "@ref": "scopus",
            "@href": "https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=85122349027&origin=inward"
          },
          {
            "@_fa": "true",
            "@ref": "scopus-citedby",
            "@href": "https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=85122349027&origin=inward"
          }
        ],
        "prism:url": "https://api.elsevier.com/content/abstract/scopus_id/85122349027",
        "dc:identifier": "SCOPUS_ID:85122349027",
        "eid": "2-s2.0-85122349027",
        "dc:title": "Evaluation of cytotoxicity, apoptosis, and angiogenesis induced by Kombucha extract-loaded PLGA nanoparticles in human ovarian cancer cell line (A2780)",
        "dc:creator": "Ghandehari S.",
        "prism:publicationName": "Biomass Conversion and Biorefinery",
        "prism:issn": "21906815",
        "prism:eIssn": "21906823",
        "prism:pageRange": null,
        "prism:coverDate": "2022-01-01",
        "prism:coverDisplayDate": "2022",
        "prism:doi": "10.1007/s13399-021-02283-2",
        "citedby-count": "0",
        "affiliation": [
          {
            "@_fa": "true",
            "affilname": "Islamic Azad University, Shahrood Branch",
            "affiliation-city": "Shahrood",
            "affiliation-country": "Iran"
          }
        ],
        "prism:aggregationType": "Journal",
        "subtype": "ar",
        "subtypeDescription": "Article",
        "source-id": "21100466851",
        "openaccess": "0",
        "openaccessFlag": null
      }
    ]
  }
}
```
