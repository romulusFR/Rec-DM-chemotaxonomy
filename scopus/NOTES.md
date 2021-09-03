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
