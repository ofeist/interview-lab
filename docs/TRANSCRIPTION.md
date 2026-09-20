# Postupak transkripcije intervjua

Skripta `scripts/transcribe_interview.py` čuva original u `data/raw/`, izrađuje
izvedeni audio u `data/work/` i sprema rezultate u `data/results/`. API ključ
čita isključivo iz varijable okruženja `OPENAI_API_KEY`; ključ se ne zapisuje u
projekt, zapis postavki ni naredbeni redak.

## Odabrani postupak

- model: `gpt-4o-transcribe-diarize`
- jezik: `de`
- format odgovora: `diarized_json`
- podjela na strani API-ja: `chunking_strategy=auto` (VAD)
- prijenos odgovora: `stream=true`; skripta prikazuje napredak po dovršenim
  govornim segmentima i izbjegava dugo čekanje bez mrežnih podataka
- govornici: neutralne oznake `S1`, `S2` itd.; uloge se dodjeljuju tek nakon
  ljudske provjere
- lokalni dijelovi: najviše približno 20 minuta, granice blizu tišine i 10
  sekundi preklapanja sa svake strane; ovo je potrebno jer je API za odabrani
  model vratio maksimalno trajanje od 1.400 sekundi po zahtjevu
- dosljednost govornika: nakon prvog dijela skripta izrađuje kratke lokalne
  glasovne reference (do četiri govornika) i šalje ih uz sljedeće dijelove;
  dodatno uspoređuje govornike u preklapanju
- spajanje: vremenske oznake vraćaju se na vrijeme originala, a segmenti se
  zadržavaju prema središtu nepreklapajućeg područja kako se preklapanje ne bi
  dupliciralo
- oporavak: svaki uspješno dovršen API dio odmah se sprema i ponovno koristi
  nakon mrežne pogreške ili prekida procesa

Službena dokumentacija:

- https://developers.openai.com/api/docs/guides/speech-to-text
- https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize
- https://developers.openai.com/api/docs/pricing

## Naredbe

Tehnički pregled bez promjena:

```bash
python3 scripts/transcribe_interview.py inspect "data/raw/IME_DATOTEKE.mp4"
```

Priprema osam minuta bez API poziva:

```bash
python3 scripts/transcribe_interview.py prepare \
  "data/raw/IME_DATOTEKE.mp4" --kind sample --duration 480
```

Probna transkripcija prvih osam minuta:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
python3 scripts/transcribe_interview.py sample \
  "data/raw/IME_DATOTEKE.mp4" --duration 480
unset OPENAI_API_KEY
```

Nakon provjere uzorka, cijeli intervju obrađuje se istom skriptom:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
python3 scripts/transcribe_interview.py full "data/raw/IME_DATOTEKE.mp4"
unset OPENAI_API_KEY
```

Ako je snimka dulja od sigurnog limita modela, naredba `full` automatski
priprema i obrađuje lokalne dijelove. Nije potrebna posebna naredba za spajanje.
Ponovno pokretanje iste naredbe koristi već dovršene dijelove. Nemojte koristiti
`--force` pri nastavku prekinutog rada jer ta opcija ponovno priprema audio i
ponovno šalje sve dijelove API-ju.

Ako ključ postavljate preko upravitelja tajni ili lokalne postavke okruženja,
izostavite `export`/`unset`. Ne stavljajte ključ u `.env`, izvornu skriptu,
shell-povijest ili razgovor bez izričite zaštite tog spremišta.

## Izlazi

Za svaki način (`sample` ili `full`) nastaju:

- `api_raw.json`: objedinjeni streaming odgovor, uključujući izvorne API
  događaje u `stream_events`;
- `transcript.json`: normalizirani segmenti, vremena u odnosu na original i
  oznake `S1`, `S2`;
- `transcript.txt`: čitljivi odlomci s govornicima i vremenskim oznakama;
- `manual_review.txt`: heuristički popis mjesta za preslušavanje;
- `run_manifest.json`: model, postavke, kontrolne sume, izvor i izlazne putanje.

Za lokalno podijeljeni intervju nastaje i poddirektorij `chunks/` s izvornim
API odgovorom svakog dijela. `manual_review.txt` automatski uključuje područja
oko svih mjesta spajanja.

`gpt-4o-transcribe-diarize` uz `diarized_json` ne vraća pouzdanost riječi
(`logprobs`). Zato je rezultat nacrt: popis za provjeru nije iscrpan, a nejasna
mjesta treba označiti s `[unverständlich]` tek nakon preslušavanja. Skripta ne
sažima niti jezično dotjeruje tekst nakon API odgovora.

## Trošak i ograničenja (provjereno 2026-09-20)

Dokumentacija navodi limit od 25 MB po datoteci. Stranica modela navodi cijene
audio-ulaza od 2,50 USD i tekstualnog izlaza od 10,00 USD na milijun tokena.
Javni cjenik ne prikazuje zaseban minutni red za diarizirani model; zato se za
planiranje koristi približna stopa `gpt-4o-transcribe` od 0,006 USD/min, a
stvarni iznos može odstupati zbog tokenizacije.

Javna dokumentacija ne navodi maksimalno trajanje pojedinačnog zahtjeva. API je
za ovu snimku vratio limit od 1.400 sekundi za
`gpt-4o-transcribe-diarize`, pa skripta ostavlja dodatnu sigurnosnu marginu.

Početna procjena prema javnoj zamjenskoj minutnoj stopi bila je približno 0,30
USD za ovu snimku, 5,40 USD za 15 sati i 0,05 USD za probnih osam minuta.
Stvarno zabilježeni trošak prvog 8-minutnog diarizacijskog zahtjeva bio je 0,10
USD (zahtjev je završio klijentskim timeoutom). Ako se ta stopa pokaže
reprezentativnom, praktična procjena iznosi približno 0,63 USD za ovu snimku i
11,25 USD za 15 sati. Neuspjeli ili ponovljeni zahtjevi mogu se dodatno
naplatiti. Cijene i stvarnu potrošnju treba provjeriti prije veće serije.

## Metodološka napomena

Prije slanja istraživačkog materijala van lokalnog računala treba potvrditi da
su privola sudionika, etičko odobrenje, ugovor o obradi podataka i pravila
ustanove usklađeni s odabranim pružateljem API-ja. Izvedene audio-datoteke i
transkripti mogu sadržavati osobne podatke te zahtijevaju istu razinu zaštite
kao original.
