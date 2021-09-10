# NOTES

## API_KEY

- id : 1
- domain : <http://unc.nc>
- name : _chemo-diversity_
- key : _7047b3a8cf46d922d5d5ca71ff531b7d_

## API

### ScienceDirect

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

